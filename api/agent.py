import asyncio
import aiohttp
import os
import sys
import logging
import time
from pathlib import Path

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableField
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

#calculator imports
import math
import cmath
import statistics
from typing import Union, List, Any
import ast
import operator
import re

# Configure logging for agent
logger = logging.getLogger(__name__)


# Constants and Configuration
OPENAI_API_KEY = SecretStr(config.OPENAI_API_KEY)
SERPAPI_API_KEY = SecretStr(config.SERPAPI_API_KEY)

logger.info("🔧 Agent configuration loaded")
logger.debug(f"🤖 Using OpenAI model: {config.OPENAI_MODEL}")
logger.debug(f"🌡️ Temperature: {config.OPENAI_TEMPERATURE}")
logger.debug(f"🎯 Max tokens: {config.OPENAI_MAX_TOKENS}")

# LLM and Prompt Setup
llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    temperature=config.OPENAI_TEMPERATURE,
    max_tokens=config.OPENAI_MAX_TOKENS,
    streaming=True,
    api_key=OPENAI_API_KEY
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

logger.info("✅ LLM initialized successfully")

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You're a helpful assistant. When answering a user's question "
        "you should first use one of the tools provided. After using a "
        "tool the tool output will be provided back to you. When you have "
        "all the information you need, you MUST use the final_answer tool "
        "to provide a final answer to the user. Use tools to answer the "
        "user's CURRENT question, not previous questions."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# we use the article object for parsing serpapi results later
class Article(BaseModel):
    title: str
    source: str
    link: str
    snippet: str

    @classmethod
    def from_serpapi_result(cls, result: dict) -> "Article":
        return cls(
            title=result["title"],
            source=result["source"],
            link=result["link"],
            snippet=result["snippet"],
        )

# Tools definition
# note: we define all tools as async to simplify later code, but only the serpapi
# tool is actually async
class SafeMathEvaluator:
    """Safe mathematical expression evaluator with restricted operations."""
    
    # Allowed operators and functions
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }
    
    # Safe mathematical functions
    functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        # Trigonometric functions
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        # Logarithmic functions
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'ln': math.log,  # Natural logarithm alias
        'exp': math.exp,
        # Other math functions
        'sqrt': math.sqrt,
        'factorial': math.factorial,
        'ceil': math.ceil,
        'floor': math.floor,
        'degrees': math.degrees,
        'radians': math.radians,
        'gcd': math.gcd,
        'lcm': math.lcm,
        # Statistics functions
        'mean': statistics.mean,
        'median': statistics.median,
        'mode': statistics.mode,
        'stdev': statistics.stdev,
        'variance': statistics.variance,
    }
    
    # Mathematical constants
    constants = {
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
    }

    def evaluate(self, node):
        """Safely evaluate mathematical expressions."""
        if isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.Name):  # Variables/constants
            if node.id in self.constants:
                return self.constants[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = self.evaluate(node.left)
            right = self.evaluate(node.right)
            op = self.operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = self.evaluate(node.operand)
            op = self.operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
        elif isinstance(node, ast.Call):  # Function calls
            func_name = node.func.id
            if func_name in self.functions:
                args = [self.evaluate(arg) for arg in node.args]
                return self.functions[func_name](*args)
            else:
                raise ValueError(f"Unknown function: {func_name}")
        elif isinstance(node, ast.List):  # Lists for statistical functions
            return [self.evaluate(item) for item in node.elts]
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def safe_eval(self, expression: str):
        """Parse and evaluate a mathematical expression safely."""
        try:
            # Remove whitespace and validate expression
            expression = expression.strip()
            if not expression:
                raise ValueError("Empty expression")
            
            # Parse the expression
            tree = ast.parse(expression, mode='eval')
            return self.evaluate(tree.body)
        except Exception as e:
            raise ValueError(f"Error evaluating expression: {str(e)}")

@tool
async def advanced_calculator(expression: str) -> str:
    """
    Advanced calculator that can evaluate complex mathematical expressions.
    
    Supports:
    - Basic arithmetic: +, -, *, /, **, %, //
    - Mathematical functions: sin, cos, tan, log, sqrt, factorial, etc.
    - Statistical functions: mean, median, mode, stdev, variance
    - Constants: pi, e, tau
    - Lists for statistical operations: mean([1,2,3,4,5])
    
    Examples:
    - "2 + 3 * 4"
    - "sqrt(16) + log(100, 10)"
    - "sin(pi/4) * cos(pi/4)"
    - "factorial(5) / (2**3)"
    - "mean([1, 2, 3, 4, 5])"
    - "log(e**2)"
    """
    logger.info(f"🧮 Advanced calculator called with expression: '{expression}'")
    start_time = time.time()
    
    try:
        evaluator = SafeMathEvaluator()
        result = evaluator.safe_eval(expression)
        
        # Format the result appropriately
        if isinstance(result, float):
            if result.is_integer():
                formatted_result = str(int(result))
            else:
                formatted_result = f"{result:.10g}"  # Remove trailing zeros
        else:
            formatted_result = str(result)
        
        execution_time = time.time() - start_time
        logger.info(f"✅ Calculator result: {formatted_result} (computed in {execution_time:.3f}s)")
        return formatted_result
            
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        logger.error(f"❌ Calculator error after {execution_time:.3f}s: {error_msg}")
        return error_msg

@tool 
async def calculate_compound_interest(
    principal: float, 
    rate: float, 
    time: float, 
    compound_frequency: int = 1
) -> str:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Time in years
        compound_frequency: Number of times interest compounds per year (default: 1)
    
    Returns formatted string with final amount and interest earned.
    """
    logger.info(f"💰 Compound interest calculation: P=${principal}, R={rate*100}%, T={time}y, F={compound_frequency}")
    
    try:
        amount = principal * (1 + rate/compound_frequency)**(compound_frequency * time)
        interest = amount - principal
        result = f"Final Amount: ${amount:.2f}, Interest Earned: ${interest:.2f}"
        logger.info(f"✅ Compound interest result: {result}")
        return result
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"❌ Compound interest error: {error_msg}")
        return error_msg

@tool
async def solve_quadratic(a: float, b: float, c: float) -> str:
    """
    Solve quadratic equation ax² + bx + c = 0.
    
    Returns the roots of the equation.
    """
    logger.info(f"📐 Quadratic equation solver: {a}x² + {b}x + {c} = 0")
    
    try:
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)
            result = f"Two real roots: x₁ = {root1:.6g}, x₂ = {root2:.6g}"
        elif discriminant == 0:
            root = -b / (2*a)
            result = f"One real root: x = {root:.6g}"
        else:
            real_part = -b / (2*a)
            imag_part = math.sqrt(-discriminant) / (2*a)
            result = f"Two complex roots: x₁ = {real_part:.6g} + {imag_part:.6g}i, x₂ = {real_part:.6g} - {imag_part:.6g}i"
        
        logger.info(f"✅ Quadratic solution: {result}")
        return result
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"❌ Quadratic solver error: {error_msg}")
        return error_msg

@tool
async def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units.
    
    Supported conversions:
    - Length: m, cm, mm, km, in, ft, yd, mile
    - Weight: kg, g, lb, oz
    - Temperature: celsius, fahrenheit, kelvin
    - Area: m2, cm2, ft2, in2
    - Volume: l, ml, gal, qt, pt, cup
    """
    logger.info(f"🔄 Unit conversion: {value} {from_unit} → {to_unit}")
    
    try:
        # Length conversions (to meters)
        length_to_meters = {
            'm': 1, 'meter': 1, 'meters': 1,
            'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01,
            'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
            'km': 1000, 'kilometer': 1000, 'kilometers': 1000,
            'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
            'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
            'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
            'mile': 1609.34, 'miles': 1609.34
        }
        
        # Weight conversions (to kg)
        weight_to_kg = {
            'kg': 1, 'kilogram': 1, 'kilograms': 1,
            'g': 0.001, 'gram': 0.001, 'grams': 0.001,
            'lb': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
            'oz': 0.0283495, 'ounce': 0.0283495, 'ounces': 0.0283495
        }
        
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Temperature conversions
        if from_unit in ['celsius', 'c'] and to_unit in ['fahrenheit', 'f']:
            result = (value * 9/5) + 32
            result_str = f"{value}°C = {result:.2f}°F"
        elif from_unit in ['fahrenheit', 'f'] and to_unit in ['celsius', 'c']:
            result = (value - 32) * 5/9
            result_str = f"{value}°F = {result:.2f}°C"
        elif from_unit in ['celsius', 'c'] and to_unit in ['kelvin', 'k']:
            result = value + 273.15
            result_str = f"{value}°C = {result:.2f}K"
        elif from_unit in ['kelvin', 'k'] and to_unit in ['celsius', 'c']:
            result = value - 273.15
            result_str = f"{value}K = {result:.2f}°C"
        
        # Length conversions
        elif from_unit in length_to_meters and to_unit in length_to_meters:
            meters = value * length_to_meters[from_unit]
            result = meters / length_to_meters[to_unit]
            result_str = f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        # Weight conversions
        elif from_unit in weight_to_kg and to_unit in weight_to_kg:
            kg = value * weight_to_kg[from_unit]
            result = kg / weight_to_kg[to_unit]
            result_str = f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        else:
            result_str = f"Error: Conversion from {from_unit} to {to_unit} not supported"
        
        logger.info(f"✅ Unit conversion result: {result_str}")
        return result_str
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logger.error(f"❌ Unit conversion error: {error_msg}")
        return error_msg

# @tool
# async def add(x: float, y: float) -> float:
#     """Add 'x' and 'y'."""
#     return x + y

# @tool
# async def multiply(x: float, y: float) -> float:
#     """Multiply 'x' and 'y'."""
#     return x * y

# @tool
# async def exponentiate(x: float, y: float) -> float:
#     """Raise 'x' to the power of 'y'."""
#     return x ** y

# @tool
# async def subtract(x: float, y: float) -> float:
#     """Subtract 'x' from 'y'."""
#     return y - x

@tool
async def serpapi(query: str) -> list[Article]:
    """Use this tool to search the web."""
    logger.info(f"🔍 SerpAPI search called with query: '{query}'")
    start_time = time.time()
    
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query,
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://serpapi.com/search",
                params=params
            ) as response:
                if response.status != 200:
                    logger.error(f"❌ SerpAPI request failed with status {response.status}")
                    return []
                
                results = await response.json()
                
        articles = [Article.from_serpapi_result(result) for result in results["organic_results"]]
        execution_time = time.time() - start_time
        
        logger.info(f"✅ SerpAPI returned {len(articles)} results in {execution_time:.3f}s")
        logger.debug(f"📝 Article titles: {[article.title[:50] + '...' if len(article.title) > 50 else article.title for article in articles[:3]]}")
        
        return articles
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ SerpAPI error after {execution_time:.3f}s: {e}")
        return []

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
    logger.info(f"🎯 Final answer generated using tools: {tools_used}")
    logger.debug(f"📝 Answer length: {len(answer)} characters")
    return {"answer": answer, "tools_used": tools_used}

# tools = [add, subtract, multiply, exponentiate, final_answer, serpapi]
tools = [
    advanced_calculator,
    calculate_compound_interest, 
    solve_quadratic,
    convert_units,
    final_answer,
    serpapi
]
# note when we have sync tools we use tool.func, when async we use tool.coroutine
name2tool = {tool.name: tool.coroutine for tool in tools}

logger.info(f"🛠️ Loaded {len(tools)} tools: {[tool.name for tool in tools]}")

# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False
        logger.debug("📡 QueueCallbackHandler initialized")

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                logger.debug("🏁 Streaming completed")
                return
            if token_or_done:
                yield token_or_done
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
                logger.debug("🎯 Final answer tool detected in stream")
        self.queue.put_nowait(kwargs.get("chunk"))
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            logger.debug("✅ LLM stream ended with final answer")
            self.queue.put_nowait("<<DONE>>")
        else:
            logger.debug("🔄 LLM step completed, continuing...")
            self.queue.put_nowait("<<STEP_END>>")

async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    
    logger.info(f"🔧 Executing tool '{tool_name}' with args: {tool_args}")
    start_time = time.time()
    
    try:
        tool_out = await name2tool[tool_name](**tool_args)
        execution_time = time.time() - start_time
        logger.info(f"✅ Tool '{tool_name}' completed in {execution_time:.3f}s")
        
        return ToolMessage(
            content=f"{tool_out}",
            tool_call_id=tool_call.tool_calls[0]["id"]
        )
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ Tool '{tool_name}' failed after {execution_time:.3f}s: {e}")
        return ToolMessage(
            content=f"Error executing {tool_name}: {str(e)}",
            tool_call_id=tool_call.tool_calls[0]["id"]
        )

# Agent Executor
class CustomAgentExecutor:
    def __init__(self, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="any")
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False) -> dict:
        logger.info(f"🚀 Agent invoked with input: '{input[:100]}{'...' if len(input) > 100 else ''}'")
        start_time = time.time()
        
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        
        # streaming function
        async def stream(query: str) -> list[AIMessage]:
            logger.debug(f"🔄 Starting stream iteration {count + 1}")
            
            response = self.agent.with_config(
                callbacks=[streamer]
            )
            # we initialize the output dictionary that we will be populating with
            # our streamed output
            outputs = []
            # now we begin streaming
            async for token in response.astream({
                "input": query,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            }):
                tool_calls = token.additional_kwargs.get("tool_calls")
                if tool_calls:
                    # first check if we have a tool call id - this indicates a new tool
                    if tool_calls[0]["id"]:
                        outputs.append(token)
                    else:
                        outputs[-1] += token
                else:
                    pass
            
            logger.debug(f"📤 Stream iteration completed with {len(outputs)} tool calls")
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]

        while count < self.max_iterations:
            logger.info(f"🔄 Agent iteration {count + 1}/{self.max_iterations}")
            
            # invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
            
            if not tool_calls:
                logger.warning(f"⚠️ No tool calls generated in iteration {count + 1}")
                break
            
            logger.info(f"🛠️ Generated {len(tool_calls)} tool calls: {[tc.tool_calls[0]['name'] for tc in tool_calls]}")
            
            # gather tool execution coroutines
            tool_obs = await asyncio.gather(
                *[execute_tool(tool_call) for tool_call in tool_calls]
            )
            # append tool calls and tool observations to the scratchpad in order
            id2tool_obs = {tool_call.tool_call_id: tool_obs for tool_call, tool_obs in zip(tool_calls, tool_obs)}
            for tool_call in tool_calls:
                agent_scratchpad.extend([
                    tool_call,
                    id2tool_obs[tool_call.tool_call_id]
                ])
            
            count += 1
            # if the tool call is the final answer tool, we stop
            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer = final_answer_call["args"]["answer"]
                    found_final_answer = True
                    logger.info(f"🎯 Final answer found: '{final_answer[:100]}{'...' if len(final_answer) > 100 else ''}'")
                    break
            
            # Only break the loop if we found a final answer
            if found_final_answer:
                break
        
        execution_time = time.time() - start_time
        
        if final_answer is None:
            logger.warning(f"⚠️ Agent reached max iterations ({self.max_iterations}) without final answer")
            final_answer = "No answer found"
            final_answer_call = {"answer": "No answer found", "tools_used": []}
        
        # add the final output to the chat history, we only add the "answer" field
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else "No answer found")
        ])
        
        logger.info(f"✅ Agent completed in {execution_time:.3f}s with {count} iterations")
        logger.debug(f"💬 Chat history now has {len(self.chat_history)} messages")
        
        # return the final answer in dict form
        return final_answer_call if final_answer else {"answer": "No answer found", "tools_used": []}

# Initialize agent executor
agent_executor = CustomAgentExecutor()
logger.info("🤖 Agent executor initialized and ready")  