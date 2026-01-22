#!/usr/bin/env python3
"""
Simple MITM Proxy Script for mitmproxy
Usage: mitmproxy -s proxy_server.py -p 8008
"""

import json
import time
from datetime import datetime
from mitmproxy import http, ctx


class TrafficLogger:
    """Log HTTP traffic passing through the proxy"""
    
    def __init__(self):
        self.request_count = 0
        self.response_count = 0
        self.start_time = time.time()
    
    def request(self, flow: http.HTTPFlow) -> None:
        """Called when a request is received"""
        self.request_count += 1
        
        method = flow.request.method
        url = flow.request.pretty_url
        headers = dict(flow.request.headers)
        
        log_data = {
            "type": "request",
            "timestamp": datetime.now().isoformat(),
            "count": self.request_count,
            "method": method,
            "url": url,
            "headers": dict(headers),
            "content_length": len(flow.request.content) if flow.request.content else 0
        }
        
        ctx.log.info(json.dumps(log_data))
    
    def response(self, flow: http.HTTPFlow) -> None:
        """Called when a response is received"""
        self.response_count += 1
        
        status = flow.response.status_code
        method = flow.request.method
        url = flow.request.pretty_url
        
        log_data = {
            "type": "response",
            "timestamp": datetime.now().isoformat(),
            "count": self.response_count,
            "status": status,
            "method": method,
            "url": url,
            "content_length": len(flow.response.content) if flow.response.content else 0
        }
        
        ctx.log.info(json.dumps(log_data))


# Create addon instance
addons = [
    TrafficLogger()
]
