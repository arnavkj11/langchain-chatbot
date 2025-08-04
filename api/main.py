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
        # Create task with timeout
        task = asyncio.create_task(agent_executor.invoke(
            input=content,
            streamer=streamer,
            verbose=True  # set to True to see verbose output in console
        ))
        
        # Add overall timeout for the entire process
        timeout_task = asyncio.create_task(asyncio.sleep(300))  # 5 minute timeout
        
        # Track streaming state
        token_count = 0
        max_tokens = 10000
        last_token_time = asyncio.get_event_loop().time()
        token_timeout = 30  # 30 seconds between tokens
        
        logger.info("Starting token streaming...")
        
        # initialize various components to stream
        try:
            async for token in streamer:
                try:
                    current_time = asyncio.get_event_loop().time()
                    
                    # Check for token timeout
                    if current_time - last_token_time > token_timeout:
                        logger.warning("Token timeout reached, ending stream")
                        break
                        
                    last_token_time = current_time
                    token_count += 1
                    
                    # Prevent infinite streaming
                    if token_count > max_tokens:
                        logger.warning("Maximum token limit reached, ending stream")
                        break
                    
                    # Log token progress periodically
                    if token_count % 100 == 0:
                        logger.debug(f"Processed {token_count} tokens")
                    
                    if token == "<<STEP_END>>":
                        logger.debug("Received STEP_END token")
                        yield "</step>"
                        
                    elif token == "<<DONE>>":
                        logger.info("Received DONE token, ending stream")
                        break
                        
                    elif hasattr(token, 'message') and token.message and token.message.additional_kwargs.get("tool_calls"):
                        tool_calls = token.message.additional_kwargs.get("tool_calls")
                        if tool_calls and len(tool_calls) > 0:
                            try:
                                if tool_name := tool_calls[0]["function"]["name"]:
                                    logger.info(f"Tool call detected: {tool_name}")
                                    yield f"<step><step_name>{tool_name}</step_name>"
                                    
                                if tool_args := tool_calls[0]["function"]["arguments"]:
                                    logger.debug(f"Tool args: {tool_args[:100]}{'...' if len(tool_args) > 100 else ''}")
                                    yield tool_args
                            except (KeyError, IndexError, TypeError) as e:
                                logger.error(f"Error processing tool call: {str(e)}")
                                continue
                    else:
                        # Handle other token types or content
                        if hasattr(token, 'content') and token.content:
                            yield token.content
                        elif isinstance(token, str):
                            yield token
                            
                except Exception as e:
                    logger.error(f"Error processing individual token: {str(e)}")
                    continue
                    
        except asyncio.TimeoutError:
            logger.error("Streaming timed out")
        except Exception as e:
            logger.error(f"Error in token streaming loop: {str(e)}")
            
        # Wait for the agent task to complete or timeout
        try:
            done, pending = await asyncio.wait(
                [task, timeout_task], 
                return_when=asyncio.FIRST_COMPLETED,
                timeout=10.0  # Additional 10 second grace period
            )
            
            # Cancel any pending tasks
            for pending_task in pending:
                pending_task.cancel()
                
            # Check if the main task completed
            if task in done:
                try:
                    result = await task
                    logger.info("Agent task completed successfully")
                except Exception as e:
                    logger.error(f"Agent task failed: {str(e)}")
            else:
                logger.warning("Agent task timed out")
                task.cancel()
                
        except asyncio.TimeoutError:
            logger.error("Final timeout reached, cancelling agent task")
            task.cancel()
        except Exception as e:
            logger.error(f"Error waiting for agent task: {str(e)}")
            
        logger.info(f"Token generation completed. Total tokens processed: {token_count}")
        
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
