import os
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import websockets

from metrics_ws import websocket_handler, monitor_cleaner
from utils.logger_config import setup_logger

from config.settings import (
    LOG_FILE_DIR,
    STATIC_DIR,
    HOST,
    HTTP_PORT,
)

# --- Set logger ---
logger = setup_logger(__name__)

# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🌱 Starting lifespan and background monitor_cleaner")
    task = asyncio.create_task(monitor_cleaner())
    try:
        yield
    finally:
        logger.info("🛑 Shutting down, cancelling monitor_cleaner")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info("✅ monitor_cleaner cancelled successfully")

app = FastAPI(lifespan=lifespan)

try:
    os.makedirs(LOG_FILE_DIR, exist_ok=True)
    app.mount("/downloads", StaticFiles(directory=LOG_FILE_DIR), name="downloads")
except Exception as e:
    logger.error(f"Error mounting downloads directory: {e}, ")
    raise RuntimeError(f"Please create the directory '{LOG_FILE_DIR}' manually.")

# get web interface
@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "preview.html"
    return index_file.read_text(encoding="utf-8")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_handler(websocket)
    
# @app.on_event("startup")
# async def startup():
#     asyncio.create_task(monitor_cleaner())

# Activate FastAPI server
if __name__ == "__main__":
    config = uvicorn.Config(app, host=HOST, port=HTTP_PORT, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())