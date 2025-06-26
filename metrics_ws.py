import asyncio
import time
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import time
from rich.live import Live
from rich.table import Table
from rich.console import Console
from rich import box
import math
from urllib3.exceptions import InsecureRequestWarning
import urllib3
import os
from dotenv import load_dotenv
import uuid
from utils.url import normalize_url
from datasets import load_dataset
import websockets
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
import httpx

from utils.logger_config import setup_logger
from pathlib import Path

from config.settings import (
    LOG_FILE_DIR,
    PROXY,
)
from visualize import APIMetricsVisualizer

logger = setup_logger(__name__)

class Template(str):
    pass

class Conversation(str):
    pass

load_dotenv()

runtime_uuid = None

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

questions = []
count_id = 0
# Global monitor instance
monitor = None
monitor_task = None
connected_clients = set()

class FileHandler:
    def __init__(self, filename: str, mode: str, virtual: bool = False):
        self.filename = filename
        self.file = open(filename, mode) if not virtual else None

    def write(self, data):
        if self.file:
            self.file.write(data)

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class APIThroughputMonitor:
    def __init__(self, model: str, api_url: str, api_key: str, max_concurrent: int = 5, columns: int = 3, log_file: str = "api_monitor.jsonl", plot_file: str = "api_metrics.png", request_log_file: str = "request_api_monitor.jsonl", output_dir: str = None):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.columns = columns
        self.log_file = log_file
        self.plot_file = plot_file
        self.sessions = {}
        self.lock = asyncio.Lock()
        self.console = Console()
        self.active_sessions = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.prev_total_chars = 0
        self.last_update_time = self.start_time
        self.update_interval = 0.25
        self.output_dir = output_dir
        self.running = True
        self._stop_requested = False
        self.websocket = None
        self.pending_messages = []
        self.request_logs = []
        self.request_log_file = request_log_file
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Initialize log file
        with open(Path(self.output_dir, self.log_file).resolve(), 'w') as f:
            f.write('')
            f.close()
            
        with open(Path(self.output_dir, self.request_log_file).resolve(), 'w') as f:
            f.write('')
            f.close()
        
    def get_session_status(self, session_id, info):
        status_style = {
            "Starting": "yellow",
            "Processing": "blue",
            "Completed": "green",
            "Failed": "red"
        }.get(info["status"], "white")

        return (
            f"{session_id:3d} | "
            f"[{status_style}]{info['status']:10}[/{status_style}] | "
            f"Time: {info['response_time'] or '-':8} | "
            f"Chars: {info['total_chars']:5} | "
            f"Chunks: {info['chunks_received']:3} | "
            f"First Token Latency: {info['first_token_latency']:3} | "
        )
        
    async def generate_status_table(self, websocket):
        table = Table(
            title="API Throughput Monitor",
            box=box.ROUNDED,
            title_style="bold magenta",
            header_style="bold cyan",
        )

        for i in range(self.columns):
            table.add_column(f"Session Group {i+1}", justify="left")

        sessions_data = {}
        
        async with self.lock:
            sorted_sessions = sorted(self.sessions.items(), key=lambda x: int(x[0]))
            num_sessions = len(sorted_sessions)
            num_rows = math.ceil(num_sessions / self.columns)

            for row_idx in range(num_rows):
                row_data = []
                for col_idx in range(self.columns):
                    session_idx = row_idx * self.columns + col_idx
                    if session_idx < len(sorted_sessions):
                        session_id, info = sorted_sessions[session_idx]
                        row_data.append(self.get_session_status(session_id, info))
                        sessions_data[session_id] = info
                    else:
                        row_data.append("")
                table.add_row(*row_data)

            elapsed_time = time.time() - self.start_time
            total_chars = sum(s["total_chars"] for s in self.sessions.values())
            total_chunks = sum(s["chunks_received"] for s in self.sessions.values())
            chars_per_sec = total_chars / elapsed_time if elapsed_time > 0 else 0

            table.add_section()
            stats_summary = (
                f"[bold cyan]Summary Stats:[/bold cyan]\n"
                f"Time: {elapsed_time:.1f}s \n"
                f"Active: {self.active_sessions} | "
                f"Total: {self.total_requests} | "
                f"Success: {self.successful_requests} | "
                f"Failed: {self.failed_requests}\n"
                f"Chars/s: {chars_per_sec:.1f} | "
                f"Total Chars: {total_chars} | "
                f"Total Chunks: {total_chunks}"
            )
            table.add_row(stats_summary)
            if self.total_requests:
                success_rate = round(self.successful_requests/self.total_requests * 100, 2)
            else:
                success_rate = 0
            
            summary_stats_to_send = {
                "time": elapsed_time,
                "active": self.active_sessions,
                "total": self.total_requests,
                "success": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": success_rate,
                "chars_per_sec": chars_per_sec,
                "total_chars": total_chars,
                "total_chunks": total_chunks
            }
            
            await safe_send(websocket, {"status": "stats_update", "data": {
                    "sessions": sessions_data,
                    "dashboard": summary_stats_to_send
                }}, monitor=self)

        return table
    
    async def log_status(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        async with self.lock:
            total_chars = sum(session["total_chars"] for session in self.sessions.values())
            chars_per_second = (total_chars - self.prev_total_chars) / (current_time - self.last_log_time)
            active_sessions = len([s for s in self.sessions.values() if s["status"] in ["Starting", "Processing"]])
            completed_sessions = len([s for s in self.sessions.values() if s["status"] == "Completed"])

            tokens_latency = [self.sessions[id]['tokens_latency'] for id in self.sessions]
            tokens_amount = [self.sessions[id]['tokens_amount'] for id in self.sessions]
            ftls = []
            try:
                ftls = [self.sessions[id]['first_token_latency'] for id in self.sessions]
            except KeyError as e:
                logger.error(e)
                ftls = []
            
            for id in self.sessions:
                self.sessions[id]['tokens_latency'] = []
                self.sessions[id]['tokens_amount'] = []

            status = {
                "timestamp": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                "elapsed_seconds": elapsed,
                "total_chars": total_chars,
                "chars_per_second": round(chars_per_second, 2),
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "total_sessions": len(self.sessions),
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "tokens_latency": tokens_latency,
                "tokens_amount": tokens_amount,
                "first_token_latencies": ftls,
            }
            
            with open(Path(self.output_dir, self.log_file).resolve(), 'a') as f:
                f.write(json.dumps(status) + '\n')

            self.prev_total_chars = total_chars
            self.last_log_time = current_time
            
            # Return status for potential use
            return status
    
    def process_stream_line(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            logger.debug(f"Line: {line}")
            if line.startswith('data: '):
                line = line[6:]

            # Handle stream completion marker
            if line.strip() == '[DONE]':
                return None

            # Parse the JSON content
            data = json.loads(line)
            
            # Extract the content from the response structure
            if 'choices' in data and len(data['choices']) > 0:
                if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                    return data['choices'][0]['delta']['content']

            return None
        except json.JSONDecodeError:
            logger.error("<<< JSON parsing error >>")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            return None

    def process_stream_info(self, line):
        try:
            # Decode the line from bytes to string if necessary
            if isinstance(line, bytes):
                line = line.decode('utf-8')

            # Remove the "data: " prefix if it exists
            data_key = 'data: '
            if line.startswith(data_key):
                line = line[len(data_key):]

            if line.strip() == '[DONE]':
                return None

            data = json.loads(line)
            elapsed_time = time.time() - self.start_time
            return {"data": data, "timestamp": time.time(), "in-time": self.duration > elapsed_time}
        except json.JSONDecodeError:
            logger.error("<<< JSON parsing error >>")
            logger.debug(f"Error processing line: {line}")
            return None
        except Exception as e:
            logger.error(f"Error processing line: {str(e)}")
            logger.debug(f"Error processing line: {line}")
            return None
    

    async def make_request(self, session_id):
        async with self.semaphore:
            global count_id
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            messages = questions[session_id % len(questions)]
            payload = {
                "model": self.model,
                "stream": True,
                "messages": messages
            }
            count_id += 1
            
            start_time = time.time()
            next_token_time = start_time
            final_response = ""

            try:
                async with self.lock:
                    self.sessions[session_id] = {
                        "status": "Starting",
                        "start_time": time.time(),
                        "response_time": None,
                        "error": None,
                        "total_chars": 0,                   # only response
                        "chunks_received": 0,               # only response
                        "tokens_latency": [],               # only response, unit: s
                        "tokens_amount": [],                # only response
                        "first_token_latency": -1,          # unit: s
                    }
                
                # Make request with SSL verification disabled
                async with httpx.AsyncClient(verify=False, timeout=180.0) as client:
                    async with client.stream("POST", f"{self.api_url}/chat/completions", headers=headers, json=payload) as response:
                        status_code = response.status_code
                        payload_record = FileHandler(f"{self.output_dir}/in_{runtime_uuid}_{session_id}.json", "w", True)
                        output_record = FileHandler(f"{self.output_dir}/out_{runtime_uuid}_{session_id}.json", "w", True)

                        payload_record.write(json.dumps(payload))
                        payload_record.close()

                        async for line in response.aiter_lines():
                            if line:
                                data = self.process_stream_info(line)
                                if data is None:
                                    break
                                output_record.write(json.dumps(data) + "\n")

                                content = data["data"]["choices"][0]["delta"].get("content", "")
                                latency = round(time.time() - next_token_time, 5)
                                final_response += content
                                async with self.lock:
                                    self.sessions[session_id]["status"] = "Processing"
                                    self.sessions[session_id]["chunks_received"] += 1
                                    self.sessions[session_id]["total_chars"] += len(content)
                                    self.sessions[session_id]["tokens_amount"].append(len(content))
                                    self.sessions[session_id]["tokens_latency"].append(latency)
                                    if self.sessions[session_id]["first_token_latency"] == -1:
                                        self.sessions[session_id]["first_token_latency"] = latency
                                    next_token_time = time.time()

                        output_record.close()

                response_time = time.time() - start_time
                async with self.lock:
                    first_token_latency = self.sessions[session_id]["first_token_latency"]
                    total_chars = self.sessions[session_id]["total_chars"]
                    duration_for_speed = response_time - first_token_latency if first_token_latency > 0 else response_time
                    chars_per_sec = round(total_chars / duration_for_speed, 3) if duration_for_speed > 0 else 0.0
                    self.sessions[session_id].update({
                        "status": "Completed",
                        "response_time": f"{response_time:.2f}s",
                        "error": None
                    })
                    self.successful_requests += 1
                    
                    log_record = {
                        "session_id": session_id,
                        "timestamp": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                        "prompt": messages[0]["content"],
                        "response": final_response,
                        "latency": round(response_time, 5),
                        "status": "success",
                        "status_code": status_code,
                        "total_chars": total_chars,
                        "chunks_received": self.sessions[session_id]["chunks_received"],
                        "first_token_latency": first_token_latency,
                        "chars_per_sec": chars_per_sec
                    }
                    self.request_logs.append(log_record)
                    # ✅ 寫入檔案
                    with open(Path(self.output_dir, self.request_log_file).resolve(), 'a') as f:
                        f.write(json.dumps(log_record) + "\n")
            except asyncio.CancelledError:
                async with self.lock:
                    self.sessions[session_id].update({
                        "status": "Cancelled",
                        "error": "Cancelled by stop",
                        "response_time": "N/A"
                    })
                    self.failed_requests += 1

                    log_record = {
                        "session_id": session_id,
                        "timestamp": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                        "latency": round(time.time() - start_time, 2),
                        "status": "cancelled",
                        "error": "Cancelled by stop",
                    }
                    self.request_logs.append(log_record)
                    with open(Path(self.output_dir, self.request_log_file).resolve(), 'a') as f:
                        f.write(json.dumps(log_record) + "\n")
                raise  # 很重要，保證取消不被吞掉
            except Exception as e:
                async with self.lock:
                    logger.error(f"Error in session {session_id}: {str(e)}")
                    self.sessions[session_id].update({
                        "status": "Failed",
                        "error": str(e),
                        "response_time": "N/A"
                    })
                    self.failed_requests += 1
                    
                    log_record = {
                        # "request_id": request_id,
                        "session_id": session_id,
                        "timestamp": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
                        "latency": round(time.time() - start_time, 5),
                        "status": "fail",
                        "error": str(e),
                    }
                    self.request_logs.append(log_record)
                    # ✅ 寫入檔案
                    with open(Path(self.output_dir, self.request_log_file).resolve(), 'a') as f:
                        f.write(json.dumps(log_record) + "\n")

            finally:
                async with self.lock:
                    self.total_requests += 1
                    self.active_sessions -= 1

    def should_update_display(self):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            return True
        return False
    
    async def run(self, websocket, duration=10):
        """Async version of run for WebSocket integration"""
        self.duration = duration
        self.websocket = websocket
        self.end_time = time.time() + duration
        session_id = 0
        self.running = True
        self._stop_requested = False
        self.tasks = []
        self.timeout_notified = False
        
        logger.info(f"🧪 Starting run loop for {duration}s with max_concurrent={self.max_concurrent}")
        
        with Live(
            await self.generate_status_table(self.websocket),
            refresh_per_second=4,
            vertical_overflow="visible",
            auto_refresh=True
        ) as live:
            try: 
                while (time.time() < self.end_time and self.running and not self._stop_requested) \
                       or self.active_sessions > 0:
                    current_time = time.time()

                    if not self.timeout_notified and time.time() >= self.end_time:
                        if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                            await safe_send(self.websocket, {
                                "status": "timeout",
                                "message": "Execution time limit reached. No new requests will be sent, but running requests will continue."
                            }, monitor=self)
                        self.timeout_notified = True
    
                    if current_time - self.last_log_time >= 1.0:
                        await self.log_status()

                    if (time.time() < self.end_time and self.running and not self._stop_requested) and self.active_sessions < self.max_concurrent:
                        session_id += 1
                        async with self.lock:
                            self.active_sessions += 1
                        task = asyncio.create_task(self.make_request(session_id))
                        self.tasks.append(task)
                    
                    if self.should_update_display():
                        if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                            live.update(await self.generate_status_table(self.websocket))
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                logger.info("Monitor task was cancelled")
            finally:
                if self.websocket and self.websocket.client_state == WebSocketState.CONNECTED:
                    live.update(await self.generate_status_table(self.websocket))
                
                logger.info("🛑 Stopping... Cancelling all running make_request tasks")
                for task in self.tasks:
                    task.cancel()
                await asyncio.gather(*self.tasks, return_exceptions=True)
                self.tasks.clear()
                
                self.pending_messages = []
                # Send log file to frontend
                file_info = {
                    "status": "file",
                    "fileName": self.log_file,
                    "fileUrl": f"/downloads/{self.log_file}"
                }
                await safe_send(self.websocket, file_info, monitor=self)
                logger.info(f"📦 Log file info sent to frontend.")
                # Send log file to frontend
                file_info = {
                    "status": "file",
                    "fileName": self.request_log_file,
                    "fileUrl": f"/downloads/{self.request_log_file}"
                }
                await safe_send(self.websocket, file_info, monitor=self)
                logger.info(f"📦 Request Log file info sent to frontend.")
                # Generate charts
                try:
                    log_file_path = Path(self.output_dir, self.log_file).resolve()
                    plot_file_path = Path(self.output_dir, self.plot_file).resolve()
                    generate_visualization(log_file_path, plot_file_path)
                    logger.info(f"📊 Generating visualization from {log_file_path} to {plot_file_path}")
                    plot_info = {
                        "status": "file",
                        "fileName": self.plot_file,
                        "fileUrl": f"/downloads/{self.plot_file}"
                    }
                    await safe_send(self.websocket, plot_info, monitor=self)
                    logger.info("📈 Visualization generated successfully.")
                except Exception as e:
                    logger.error(f"❌ Failed to generate visualization: {e}")
                # Clean up states
                self.running = False
                self.timeout_notified = False
                self._stop_requested = False
                logger.info("🛑 run() has ended (timeout or stopped).")
                # logger.info(f"File Path: {file_info}")

    async def stop_monitor(self):
        self.running = False
        self._stop_requested = True


def load_dataset_as_questions(dataset_name: str, key: Template | Conversation):
    dataset = load_dataset(dataset_name)['train']
    ret = []
    if isinstance(key, Template):
        ret = []
        for row in dataset:
            conv = [
                {"role": "user", "content": key.format(**row)},
            ]
            ret.append(conv)
    elif isinstance(key, Conversation):
        for row in dataset:
            try:
                if isinstance(row[key], dict) or isinstance(row[key], list):
                    messages = row[key]
                else:
                    messages = json.loads(row[key])
                for turn in messages:
                    if 'role' in turn and 'content' in turn and isinstance(turn['role'], str) and isinstance(turn['content'], str):
                        ret.append(messages)
                    else:
                        raise ValueError(f"Invalid conversation context")
            except json.JSONDecodeError as e:
                raise ValueError(f"Can not load columns '{key}' as Conversation template")
    else:
        ret = None
    return ret

async def safe_send(websocket: WebSocket, data: dict, monitor):
    try:
        await websocket.send_text(json.dumps(data))
    except WebSocketDisconnect:
        logger.warning("⚠️ WebSocketDisconnect while sending data")
        if monitor:
            logger.info("Save message to pending message.")
            monitor.pending_messages.append(data)
            logger.info(f"P: {monitor.pending_messages}")
    except Exception as e:
        logger.error(f"❌ Unexpected error during send_text: {e}")
        if monitor:
            monitor.pending_messages.append(data)

async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    global monitor, monitor_task, connected_clients, count_id, questions
    
    client_ip = websocket.client.host
    client_port = websocket.client.port
    logger.info(f"Client connected: {client_ip}:{client_port}")
    connected_clients.add(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                logger.info(f"Failed to parse message: {e}")
                continue

            # 處理 "start" 命令
            if data.get("command") == "start":
                runtime_uuid = str(uuid.uuid4()).replace("-", "")
                if monitor and monitor.running:
                    await safe_send(websocket, {"status": "error", "message": "Monitor already running"}, monitor=monitor)
                    logger.info(f"Monitor already running: {monitor.sessions}")
                else:
                    params = data.get("params", {})
                    model = params.get('model', os.getenv('MODEL', 'gpt-3.5-turbo'))
                    api_url = normalize_url(params.get('api_url', os.environ.get('API_URL')))
                    api_key = params.get('api_key', os.environ.get('OPENAI_API_KEY'))
                    time_limit = int(params.get('time_limit', 10))
                    max_concurrent = int(params.get('max_concurrent', 5))
                    columns = int(params.get('columns', 3))
                    # Result File
                    def get_value_or_default(value: str | None, default: str) -> str:
                        if value is None or value.strip() == "":
                            return default
                        return value
                                                         
                    taipei_time = datetime.now(ZoneInfo("Asia/Taipei"))
                    current_time = taipei_time.strftime("%Y%m%d_%H%M%S")
                    output_dir = get_value_or_default(params.get("output_dir"), LOG_FILE_DIR)    # Load it from environment variables
                    log_file = get_value_or_default(params.get("log_file"), f"api_monitor_{current_time}.jsonl")
                    plot_file = get_value_or_default(params.get("plot_file"), f"api_metrics_{current_time}.png")
                    request_log_file = get_value_or_default(params.get("request_log_file"), f"request_api_monitor_{current_time}.jsonl")
                    # Datasets
                    dataset_name = params.get('dataset', "tatsu-lab/alpaca") # Dangours, use with caution
                    template_str = params.get('template')
                    conversation_str = params.get('conversation')
                    # log level 
                    console_log_level=params.get("console_log_level")
                    file_log_level=params.get("file_log_level")
                    setup_logger(__name__, console_log_level, file_log_level)
                    logger.info(f"Log level updated via websocket command: CLL: {console_log_level}, FLL: {file_log_level}")

                    # Create directories if needed
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                        
                    if monitor:
                        await monitor.stop_monitor()

                    async def start_monitor():
                        try:
                            global monitor, monitor_task, questions
                            # Load dataset
                            logger.info(f"Loading dataset '{dataset_name}' with template '{template_str}' or conversation '{conversation_str}'")
                            if template_str is not None and template_str != "":
                                questions = load_dataset_as_questions(dataset_name, Template(template_str))
                            elif conversation_str is not None and conversation_str != "":
                                questions = load_dataset_as_questions(dataset_name, Conversation(conversation_str))
                            else:
                                await safe_send(websocket, {"status": "error", "message": "Either template or conversation must be provided"}, monitor=monitor)
                                return

                            monitor = APIThroughputMonitor(
                                model=model,
                                api_url=api_url,
                                api_key=api_key,
                                max_concurrent=max_concurrent,
                                columns=columns,
                                log_file=log_file,
                                plot_file=plot_file,
                                request_log_file=request_log_file,
                                output_dir=output_dir
                            )
                            
                            # Start the monitor
                            logger.info("🚀 Starting API Throughput Monitor...")
                            try:
                                await safe_send(websocket, {"status": "started", "message": "Monitor started"}, monitor=monitor)
                            except Exception:
                                logger.warning("⚠️ Cannot notify frontend, websocket disconnected before start.")
                            await monitor.run(websocket, duration=time_limit)
                        except Exception as e:
                            logger.exception(f"❌ Monitor run failed during startup or execution: {e}")
                # Run the monitor in the background
                monitor_task = asyncio.create_task(start_monitor())
            elif data.get("command") == "stop":
                if monitor and monitor.running:
                    await monitor.stop_monitor()
                    logger.info("Stop signal sent to monitor. Waiting for it to finish naturally.")
                    if monitor_task:
                        monitor_task.cancel()
                        try:
                            await monitor_task
                        except asyncio.CancelledError:
                            logger.info("Monitor task cancelled successfully.")

                    monitor = None
                    monitor_task = None
                    count_id = 0
                    await safe_send(websocket, {"status": "stopping", "message": "Monitor stopping"}, monitor=monitor)

                else:
                    await safe_send(websocket, {"status": "error", "message": "No monitor running"}, monitor=monitor)
            elif data.get("command") == "rebind":
                if monitor:
                    monitor.websocket = websocket
                    await safe_send(websocket, {"status": "rebound", "message": "WebSocket rebound to monitor", "running": monitor.running}, monitor=monitor)
                    logger.info(f"🔁 WebSocket rebound from client {client_ip}:{client_port}")

                    # 補發未送出的訊息
                    for msg in monitor.pending_messages:
                        try:
                            await safe_send(websocket, msg, monitor=monitor)
                            logger.info(f"📤 Resent pending message: {msg}")
                        except Exception as e:
                            logger.error(f"❌ Failed to resend pending message: {e}")

                    # 清空已發送的訊息
                    monitor.pending_messages.clear()
                else:
                    await safe_send(websocket, {"status": "no_monitor", "message": "No active monitor to rebind"}, monitor=monitor)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_ip}:{client_port}")
    except json.JSONDecodeError:
        logger.error("Invalid JSON received")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        connected_clients.discard(websocket)
        logger.info("Monitor Task Done (after disconnect or error)")

"""Background task to clean up monitor after it finishes"""     
async def monitor_cleaner():
    try: 
        global monitor, monitor_task, count_id, connected_clients

        while True:
            if monitor_task is not None and monitor_task.done() and monitor.pending_messages == []:
                monitor = None
                monitor_task = None
                count_id = 0
                logger.info("✅ Monitor task finished, cleaning up")
                
                # Send completion message to all connected clients
                for websocket in connected_clients:
                    try:
                        client_ip = websocket.client.host
                        client_port = websocket.client.port
                        completion_message = {
                            "status": "completed",
                            "message": "Benchmark run finished"
                        }
                        await safe_send(websocket, completion_message, monitor=monitor)
                        logger.info(f"📡 Sent completion message to {client_ip}:{client_port}")
                    except Exception as e:
                        logger.error(f"Error sending message to {client_ip}:{client_port}: {str(e)}")
                        
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.exception("❌ monitor_cleaner 發生未預期錯誤：", exc_info=e)

        
def generate_visualization(log_file, plot_file):
    visualizer = APIMetricsVisualizer(log_file)
    visualizer.create_visualization(plot_file)