import asyncio
import sys
from pathlib import Path
import logging

# Add the parent directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config

from agent import QueueCallbackHandler, agent_executor
from fastapi import FastAPI, Form
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
    allow_origins=[config.FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],  # Allow multiple frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# streaming function
async def token_generator(content: str, streamer: QueueCallbackHandler):
    logger.info(f"Starting token generation for content: '{content[:50]}{'...' if len(content) > 50 else ''}'")
    
    try:
        logger.info("Creating agent executor task...")
        task = asyncio.create_task(agent_executor.invoke(
            input=content,
            streamer=streamer,
            verbose=True  # set to True to see verbose output in console
        ))
        
        logger.info("Starting token streaming...")
        token_count = 0
        
        # initialize various components to stream
        async for token in streamer:
            try:
                token_count += 1
                logger.debug(f"Processing token #{token_count}: {type(token)}")
                
                if token == "<<STEP_END>>":
                    logger.info("Received STEP_END token")
                    # send end of step token
                    yield "</step>"
                elif hasattr(token, 'message') and token.message and (tool_calls := token.message.additional_kwargs.get("tool_calls")):
                    logger.info("Processing tool call token")
                    if tool_name := tool_calls[0]["function"]["name"]:
                        logger.info(f"Tool call detected: {tool_name}")
                        # send start of step token followed by step name tokens
                        yield f"<step><step_name>{tool_name}</step_name>"
                    if tool_args := tool_calls[0]["function"]["arguments"]:
                        logger.debug(f"Tool args: {tool_args[:100]}{'...' if len(tool_args) > 100 else ''}")
                        # tool args are streamed directly, ensure it's properly encoded
                        yield tool_args
                else:
                    # Handle other token types or content
                    if hasattr(token, 'content') and token.content:
                        logger.debug("Yielding token content")
                        yield token.content
                    elif isinstance(token, str):
                        logger.debug("Yielding string token")
                        yield token
                        
            except Exception as e:
                logger.error(f"Error streaming token: {e}")
                continue
                
        logger.info("Token streaming completed, waiting for agent task...")
        await task
        logger.info(f"Agent task completed. Total tokens processed: {token_count}")
        
    except Exception as e:
        logger.error(f"Critical error in token generator: {str(e)}")
        yield f"Error: {str(e)}"

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
async def invoke(content: str = Form(...)):
    request_start_time = asyncio.get_event_loop().time()
    logger.info(f"📥 Received invoke request: '{content[:100]}{'...' if len(content) > 100 else ''}'")
    
    try:
        # Validate input
        if not content or not content.strip():
            logger.warning("Empty content received")
            return {"error": "Content cannot be empty"}
        
        # Create queue with size limit to prevent memory issues
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        streamer = QueueCallbackHandler(queue)
        
        logger.info(f"🚀 Starting streaming response for content length: {len(content)}")
        
        # return the streaming response with timeout
        async def safe_token_generator():
            try:
                async for token in token_generator(content, streamer):
                    yield token
            except Exception as e:
                logger.error(f"Error in token generator: {str(e)}")
                yield f"data: Error: {str(e)}\n\n"
                
        return StreamingResponse(
            safe_token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - request_start_time
        logger.error(f"❌ Error in invoke endpoint after {execution_time:.2f}s: {str(e)}")
        
        # Return error response instead of raising
        return {
            "error": f"Internal server error: {str(e)}",
            "execution_time": execution_time
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)