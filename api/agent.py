import asyncio
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
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            else:
                return f"{result:.10g}"  # Remove trailing zeros
        else:
            return str(result)
            
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
            return f"{value}°C = {result:.2f}°F"
        elif from_unit in ['fahrenheit', 'f'] and to_unit in ['celsius', 'c']:
            result = (value - 32) * 5/9
            return f"{value}°F = {result:.2f}°C"
        elif from_unit in ['celsius', 'c'] and to_unit in ['kelvin', 'k']:
            result = value + 273.15
            return f"{value}°C = {result:.2f}K"
        elif from_unit in ['kelvin', 'k'] and to_unit in ['celsius', 'c']:
            result = value - 273.15
            return f"{value}K = {result:.2f}°C"
        
        # Length conversions
        elif from_unit in length_to_meters and to_unit in length_to_meters:
            meters = value * length_to_meters[from_unit]
            result = meters / length_to_meters[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        # Weight conversions
        elif from_unit in weight_to_kg and to_unit in weight_to_kg:
            kg = value * weight_to_kg[from_unit]
            result = kg / weight_to_kg[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        else:
            return f"Error: Conversion from {from_unit} to {to_unit} not supported"
            
    except Exception as e:
        return f"Error: {str(e)}"

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
    params = {
        "api_key": SERPAPI_API_KEY.get_secret_value(),
        "engine": "google",
        "q": query,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://serpapi.com/search",
            params=params
        ) as response:
            results = await response.json()
    return [Article.from_serpapi_result(result) for result in results["organic_results"]]

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
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

    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            if token_or_done:
                yield token_or_done
    
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))
    
    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
            logger.info("Final answer seen")
        else:
            self.queue.put_nowait("<<STEP_END>>")
            logger.info("Step end token sent") 

async def execute_tool(tool_call: AIMessage) -> ToolMessage:
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = await name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
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
        # invoke the agent but we do this iteratively in a loop until
        # reaching a final answer
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        # streaming function
        async def stream(query: str) -> list[AIMessage]:
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
            return [
                AIMessage(
                    content=x.content,
                    tool_calls=x.tool_calls,
                    tool_call_id=x.tool_calls[0]["id"]
                ) for x in outputs
            ]

        while count < self.max_iterations:
            # invoke a step for the agent to generate a tool call
            tool_calls = await stream(query=input)
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
                    break
            
            # Only break the loop if we found a final answer
            if found_final_answer:
                break
            
        # add the final output to the chat history, we only add the "answer" field
        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else "No answer found")
        ])
        # return the final answer in dict form
        return final_answer_call if final_answer else {"answer": "No answer found", "tools_used": []}

# Initialize agent executor
agent_executor = CustomAgentExecutor()  