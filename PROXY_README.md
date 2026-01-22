# MITM Proxy Server

Simple MITM proxy script using mitmproxy.

## Installation

```bash
pip install mitmproxy
```

## Usage

### Start the proxy on port 8008

```bash
mitmproxy -s proxy_server.py -p 8008
```

### For silent mode with logging only

```bash
mitmdump -s proxy_server.py -p 8008
```

### Configure your application to use the proxy

```bash
export HTTP_PROXY=http://127.0.0.1:8008
export HTTPS_PROXY=http://127.0.0.1:8008
```

### Run your application

```bash
python your_app.py
```

All traffic will be logged to the proxy output.

## Options

- `-s` - Script to load
- `-p` - Port to listen on
- `-b` - Bind address (default: 127.0.0.1)
- `-q` - Quiet mode
- `-v` - Verbose mode

## Examples

```bash
# Listen on all interfaces
mitmproxy -s proxy_server.py -b 0.0.0.0 -p 8008

# Listen on specific IP
mitmproxy -s proxy_server.py -b 192.168.1.100 -p 8008

# Run with verbosity
mitmproxy -s proxy_server.py -p 8008 -v

# Headless mode
mitmdump -s proxy_server.py -p 8008
```

## What it does

- Logs all HTTP requests with method, URL, headers, and content size
- Logs all HTTP responses with status code, URL, and content size
- Outputs in JSON format for easy parsing

## Example Output

```
{"type": "request", "timestamp": "2026-01-22T14:30:45.123456", "count": 1, "method": "GET", "url": "https://api.example.com/data", "content_length": 0}
{"type": "response", "timestamp": "2026-01-22T14:30:45.456789", "count": 1, "status": 200, "url": "https://api.example.com/data", "content_length": 1234}
```
