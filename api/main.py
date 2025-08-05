import asyncio
import sys
import logging
import time
import uuid
from pathlib import Path

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

from agent import QueueCallbackHandler, agent_executor
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot_api.log')
    ]
)
logger = logging.getLogger(__name__)

# initilizing our application
app = FastAPI(title="LangChain Chatbot API", version="1.0.0")

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Request started")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"[{request_id}] {request.method} {request.url.path} - Completed in {process_time:.3f}s with status {response.status_code}")
    
    return response

# Validate configuration on startup
@app.on_event("startup")
async def validate_config():
    logger.info("🚀 Starting LangChain Chatbot API...")
    if not config.validate_config():
        logger.error("❌ Configuration validation failed")
        raise RuntimeError("Invalid configuration. Please check your .env file.")
    logger.info("✅ Configuration validated successfully")
    logger.info("🎯 API server ready to accept requests")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_URL],  # Use config for frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

logger.info(f"🔧 CORS configured for frontend URL: {config.FRONTEND_URL}")

# streaming function
async def token_generator(content: str, streamer: QueueCallbackHandler):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 🎬 Starting token generation for query: '{content[:100]}{'...' if len(content) > 100 else ''}'")
    
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True  # set to True to see verbose output in console
    ))
    
    token_count = 0
    step_count = 0
    
    # initialize various components to stream
    async for token in streamer:
        try:
            token_count += 1
            if token == "<<STEP_END>>":
                step_count += 1
                logger.debug(f"[{request_id}] 📝 Step {step_count} completed")
                # send end of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    logger.info(f"[{request_id}] 🔧 Tool '{tool_name}' started")
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                if tool_args := tool_calls[0]["function"]["arguments"]:
                    logger.debug(f"[{request_id}] 📤 Streaming tool arguments")
                    # tool args are streamed directly, ensure it's properly encoded
                    yield tool_args
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error streaming token: {e}")
            continue
    
    await task
    logger.info(f"[{request_id}] ✅ Token generation completed. Total tokens: {token_count}, Steps: {step_count}")

# Health check endpoint
@app.get("/")
async def root():
    logger.info("🏠 Root endpoint accessed")
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
    logger.info("💓 Health check endpoint accessed")
    return {
        "status": "healthy",
        "message": "LangChain Chatbot API is running",
        "config": config.get_safe_config()
    }

# invoke function
@app.post("/invoke")
async def invoke(content: str):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] 🚀 Invoke endpoint called with content length: {len(content)}")
    
    try:
        queue: asyncio.Queue = asyncio.Queue()
        streamer = QueueCallbackHandler(queue)
        
        logger.debug(f"[{request_id}] 🔄 Setting up streaming response")
        
        # return the streaming response
        response = StreamingResponse(
            token_generator(content, streamer),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
        logger.info(f"[{request_id}] 📡 Streaming response initialized")
        return response
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Error in invoke endpoint: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    import uvicorn
    logger.info("🌟 Starting server in development mode")
    uvicorn.run(app, host="0.0.0.0", port=8000)