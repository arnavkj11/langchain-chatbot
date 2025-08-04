import asyncio
import sys
from pathlib import Path
import logging

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

from agent import QueueCallbackHandler, agent_executor
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# initilizing our application
app = FastAPI(title="LangChain Chatbot API", version="1.0.0")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Validate configuration on startup
@app.on_event("startup")
async def validate_config():
    if not config.validate_config():
        raise RuntimeError("Invalid configuration. Please check your .env file.")
    # print("✅ Configuration validated successfully")
    # print("🚀 LangChain Chatbot API starting...")
    logger.info("Configuration validated successfully")
    logger.info("LangChain Chatbot API starting...")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL],  # Use config for frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True  # set to True to see verbose output in console
    ))
    # initialize various components to stream
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # send end of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                    logger.info(f"Tool call: {tool_name}") #log
                if tool_args := tool_calls[0]["function"]["arguments"]:
                    # tool args are streamed directly, ensure it's properly encoded
                    yield tool_args
                    # logger.info(f"Tool args: {tool_args}")
        except Exception as e:
            logger.error(f"Error streaming token: {e}")
            #print(f"Error streaming token: {e}")
            continue
    await task
    logger.info("Streaming completed") #log

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "🤖 LangChain Chatbot API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "chat": "/invoke",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "LangChain Chatbot API is running",
        "config": config.get_safe_config()
    }

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    try:
        queue: asyncio.Queue = asyncio.Queue()
        streamer = QueueCallbackHandler(queue)
        logger.info(f"Received content: {content}") #log
        # return the streaming response
        return StreamingResponse(
            token_generator(content, streamer),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Error in invoke endpoint: {e}") #log
        #print(f"Error in invoke endpoint: {e}")
        raise
