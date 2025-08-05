import asyncio
import logging
import aiohttp
import os
import sys
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Constants and Configuration
OPENAI_API_KEY = SecretStr(config.OPENAI_API_KEY)
SERPAPI_API_KEY = SecretStr(config.SERPAPI_API_KEY)

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
    try:
        evaluator = SafeMathEvaluator()
        result = evaluator.safe_eval(expression)
        
        # Format the result appropriately
        # Format the result appropriately
        if isinstance(result, float):
            if result.is_integer():
                formatted_result = str(int(result))
            else:
                formatted_result = f"{result:.10g}"  # Remove trailing zeros
        else:
            formatted_result = str(result)

        logger.info(f"Calculated result for '{expression}': {formatted_result}")
        return formatted_result
        

    except Exception as e:
        return f"Error: {str(e)}"

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
    try:
        amount = principal * (1 + rate/compound_frequency)**(compound_frequency * time)
        interest = amount - principal
        return f"Final Amount: ${amount:.2f}, Interest Earned: ${interest:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
async def solve_quadratic(a: float, b: float, c: float) -> str:
    """
    Solve quadratic equation ax² + bx + c = 0.
    
    Returns the roots of the equation.
    """
    try:
        discriminant = b**2 - 4*a*c
        
        if discriminant > 0:
            root1 = (-b + math.sqrt(discriminant)) / (2*a)
            root2 = (-b - math.sqrt(discriminant)) / (2*a)
            return f"Two real roots: x₁ = {root1:.6g}, x₂ = {root2:.6g}"
        elif discriminant == 0:
            root = -b / (2*a)
            return f"One real root: x = {root:.6g}"
        else:
            real_part = -b / (2*a)
            imag_part = math.sqrt(-discriminant) / (2*a)
            return f"Two complex roots: x₁ = {real_part:.6g} + {imag_part:.6g}i, x₂ = {real_part:.6g} - {imag_part:.6g}i"
    except Exception as e:
        return f"Error: {str(e)}"

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
    try:
        logger.info(f"Converting {value} from {from_unit} to {to_unit}")
        
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
            logger.info(f"Temperature conversion result: {result:.2f}°F")
            return f"{value}°C = {result:.2f}°F"
        elif from_unit in ['fahrenheit', 'f'] and to_unit in ['celsius', 'c']:
            result = (value - 32) * 5/9
            logger.info(f"Temperature conversion result: {result:.2f}°C")
            return f"{value}°F = {result:.2f}°C"
        elif from_unit in ['celsius', 'c'] and to_unit in ['kelvin', 'k']:
            result = value + 273.15
            logger.info(f"Temperature conversion result: {result:.2f}K")
            return f"{value}°C = {result:.2f}K"
        elif from_unit in ['kelvin', 'k'] and to_unit in ['celsius', 'c']:
            result = value - 273.15
            logger.info(f"Temperature conversion result: {result:.2f}°C")
            return f"{value}K = {result:.2f}°C"
        
        # Length conversions
        elif from_unit in length_to_meters and to_unit in length_to_meters:
            meters = value * length_to_meters[from_unit]
            result = meters / length_to_meters[to_unit]
            logger.info(f"Length conversion result: {result:.6g} {to_unit}")
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        # Weight conversions
        elif from_unit in weight_to_kg and to_unit in weight_to_kg:
            kg = value * weight_to_kg[from_unit]
            result = kg / weight_to_kg[to_unit]
            logger.info(f"Weight conversion result: {result:.6g} {to_unit}")
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        else:
            error_msg = f"Conversion from {from_unit} to {to_unit} not supported"
            logger.warning(error_msg)
            return f"Error: {error_msg}"
            
    except Exception as e:
        error_msg = f"Error in unit conversion: {str(e)}"
        logger.error(error_msg)
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
    try:
        logger.info(f"Starting web search for: '{query}'")
        
        params = {
            "api_key": SERPAPI_API_KEY.get_secret_value(),
            "engine": "google",
            "q": query,
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    "https://serpapi.com/search",
                    params=params
                ) as response:
                    if response.status != 200:
                        error_msg = f"SerpAPI returned status {response.status}"
                        logger.error(error_msg)
                        return []
                    
                    results = await response.json()
                    
                    # Check if we have organic results
                    if "organic_results" not in results:
                        logger.warning("No organic results found in SerpAPI response")
                        return []
                    
                    articles = [Article.from_serpapi_result(result) for result in results["organic_results"]]
                    logger.info(f"Successfully retrieved {len(articles)} search results")
                    return articles
                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error during SerpAPI request: {str(e)}")
                return []
                
    except Exception as e:
        logger.error(f"Unexpected error in SerpAPI search: {str(e)}")
        return []


@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
    logger.info(f"Providing final answer: {answer}")  # Log the final answer
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

# Streaming Handler
class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False
        self.token_count = 0
        self.max_tokens = 10000  # Prevent infinite streaming
        self._done = False
        logger.debug("QueueCallbackHandler initialized")

    def __aiter__(self):
        logger.debug("Starting streaming iteration")
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
            
        while True:
            try:
                if self.queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                token_or_done = await self.queue.get()
                
                if token_or_done == "<<DONE>>":
                    logger.info(f"Streaming completed. Total tokens: {self.token_count}")
                    self._done = True
                    raise StopAsyncIteration
                    
                if token_or_done:
                    self.token_count += 1
                    if self.token_count > self.max_tokens:
                        logger.warning("Maximum token limit reached, stopping stream")
                        self._done = True
                        raise StopAsyncIteration
                    return token_or_done
                    
            except Exception as e:
                logger.error(f"Error in streaming iteration: {str(e)}")
                self._done = True
                raise StopAsyncIteration
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        try:
            chunk = kwargs.get("chunk")
            if chunk and chunk.message and chunk.message.additional_kwargs.get("tool_calls"):
                tool_calls = chunk.message.additional_kwargs["tool_calls"]
                if tool_calls and len(tool_calls) > 0:
                    tool_name = tool_calls[0]["function"]["name"]
                    logger.debug(f"Tool call detected: {tool_name}")
                    if tool_name == "final_answer":
                        self.final_answer_seen = True
                        logger.info("Final answer tool detected")
            
            # Safely add to queue
            try:
                self.queue.put_nowait(kwargs.get("chunk"))
            except asyncio.QueueFull:
                logger.warning("Queue is full, dropping token")
                
        except Exception as e:
            logger.error(f"Error in on_llm_new_token: {str(e)}")
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        try:
            if self.final_answer_seen:
                self.queue.put_nowait("<<DONE>>")
                logger.info("Final answer seen, sending DONE signal")
            else:
                self.queue.put_nowait("<<STEP_END>>")
                logger.debug("Step completed, sending STEP_END signal")
        except Exception as e:
            logger.error(f"Error in on_llm_end: {str(e)}")
            # Force done signal in case of error
            try:
                self.queue.put_nowait("<<DONE>>")
            except:
                pass 

async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    try:
        tool_name = tool_call.tool_calls[0]["name"]
        tool_args = tool_call.tool_calls[0]["args"]
        tool_call_id = tool_call.tool_calls[0]["id"]
        
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        # Check if tool exists
        if tool_name not in name2tool:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(error_msg)
            return ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=tool_call_id
            )
        
        # Add tool execution
        try:
            tool_out = await name2tool[tool_name](**tool_args)
            logger.info(f"Successfully executed tool: {tool_name}")
            
            return ToolMessage(
                content=f"{tool_out}",
                tool_call_id=tool_call_id
            )
            
        except KeyError as e:
            error_msg = f"Missing required field in tool call: {str(e)}"
            logger.error(error_msg)
            return ToolMessage(
                content=f"Error: {error_msg}",
                tool_call_id=getattr(tool_call, 'tool_call_id', 'unknown')
            )
    except Exception as e:
        error_msg = f"Unexpected error executing tool: {str(e)}"
        logger.error(error_msg)
        return ToolMessage(
            content=f"Error: {error_msg}",
            tool_call_id=getattr(tool_call, 'tool_call_id', 'unknown')
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
        logger.info(f"Starting agent execution with input: {input[:100]}{'...' if len(input) > 100 else ''}")
        
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        
        # streaming function with error handling
        async def stream(query: str) -> list[AIMessage]:
            try:
                logger.debug(f"Starting streaming for iteration {count + 1}")
                response = self.agent.with_config(
                    callbacks=[streamer]
                )
                # we initialize the output dictionary that we will be populating with
                # our streamed output
                outputs = []
                
                try:
                    # Step 1: Get the async iterator directly (no await needed)
                    stream = response.astream({
                        "input": query,
                        "chat_history": self.chat_history,
                        "agent_scratchpad": agent_scratchpad
                    })

                    # Step 2: now iterate over the stream
                    async for token in stream:
                        tool_calls = token.additional_kwargs.get("tool_calls")
                        if tool_calls:
                            # New tool call starts when tool_calls[0]["id"] is truthy
                            if tool_calls[0]["id"]:
                                outputs.append(token)
                            else:
                                # Continuation of previous tool call
                                if outputs:
                                    outputs[-1] += token
                        # else: no tool calls in this token, skip or handle other tokens here

                except Exception as e:
                    logger.error(f"Error during streaming: {str(e)}")
                    return []
                    
                logger.debug(f"Streaming completed with {len(outputs)} outputs")
                
                # Safely create AIMessage objects
                result = []
                for x in outputs:
                    try:
                        if x.tool_calls and len(x.tool_calls) > 0:
                            result.append(AIMessage(
                                content=x.content,
                                tool_calls=x.tool_calls
                            ))
                    except Exception as e:
                        logger.error(f"Error creating AIMessage: {str(e)}")
                        continue
                        
                return result
                
            except Exception as e:
                logger.error(f"Error in streaming function: {str(e)}")
                return []

        try:
            while count < self.max_iterations:
                logger.info(f"Starting iteration {count + 1}/{self.max_iterations}")
                
                # invoke a step for the agent to generate a tool call
                tool_calls = await stream(query=input)
                
                if not tool_calls:
                    logger.warning("No tool calls generated, ending execution")
                    break
                
                logger.info(f"Generated {len(tool_calls)} tool calls")
                
                # gather tool execution coroutines with error handling
                try:
                    tool_obs = await asyncio.gather(
                        *[execute_tool(tool_call) for tool_call in tool_calls],
                        return_exceptions=True
                    )
                    
                    # Filter out exceptions and log them
                    valid_tool_obs = []
                    for i, obs in enumerate(tool_obs):
                        if isinstance(obs, Exception):
                            logger.error(f"Tool execution failed: {str(obs)}")
                            # Create error message
                            valid_tool_obs.append(ToolMessage(
                                content=f"Error: {str(obs)}",
                                tool_call_id=tool_calls[i].tool_calls[0]["id"] if i < len(tool_calls) else "unknown"
                            ))
                        else:
                            valid_tool_obs.append(obs)
                    
                    tool_obs = valid_tool_obs
                    
                except Exception as e:
                    logger.error(f"Error in tool execution gather: {str(e)}")
                    break
                
                # append tool calls and tool observations to the scratchpad in order
                id2tool_obs = {tool_call.tool_calls[0]["id"]: tool_obs[i] for i, tool_call in enumerate(tool_calls)}
                for tool_call in tool_calls:
                    agent_scratchpad.extend([
                        tool_call,
                        id2tool_obs[tool_call.tool_calls[0]["id"]]
                    ])
                
                count += 1
                
                # if the tool call is the final answer tool, we stop
                found_final_answer = False
                for tool_call in tool_calls:
                    try:
                        if tool_call.tool_calls[0]["name"] == "final_answer":
                            final_answer_call = tool_call.tool_calls[0]
                            final_answer = final_answer_call["args"]["answer"]
                            found_final_answer = True
                            logger.info("Final answer received")
                            break
                    except (KeyError, IndexError) as e:
                        logger.error(f"Error processing tool call: {str(e)}")
                        continue
                
                # Only break the loop if we found a final answer
                if found_final_answer:
                    break
            
            # add the final output to the chat history, we only add the "answer" field
            self.chat_history.extend([
                HumanMessage(content=input),
                AIMessage(content=final_answer if final_answer else "No answer found")
            ])
            
            # return the final answer in dict form
            logger.info("Agent execution completed")
            if final_answer:
                return final_answer_call
            else:
                return {"answer": "No answer found", "tools_used": []}
                
        except Exception as e:
            logger.error(f"Critical error in agent execution: {str(e)}")
            return {"answer": f"Error: {str(e)}", "tools_used": []}
    

# Initialize agent executor
agent_executor = CustomAgentExecutor()  