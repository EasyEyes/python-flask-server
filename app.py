import os
import tracemalloc

import psutil

from flask import Flask, request, make_response
from flask_cors import CORS, cross_origin
from impulse_response import run_ir_task

app = Flask(__name__)
CORS(app)

process = psutil.Process(os.getpid())
tracemalloc.start()

def run_impulse_response_task(request_json):
    if "payload" not in request_json:
        return 400, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return 400, "Request Body us missing a 'sample-rate' entry"
    if "P" not in request_json:
        return 400, "Request Body us missing a 'P' entry"
    
    recordedSignals = request_json["payload"]
    sampleRate = request_json["sample-rate"]
    P = request_json["P"]
    result = run_ir_task(recordedSignals, P, sampleRate)
    
    return 200, {
        "inverted-impulse-response": result
    }

def run_volume_task(request_json):
    return ''

SUPPORTED_TASKS = {
    'impulse-response': run_impulse_response_task,
    'volume': run_volume_task
}

@app.route("/task/<string:task>", methods=['POST'])
@cross_origin()
def task_handler(task):
    if task not in SUPPORTED_TASKS:
        return 'ERROR'
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        headers = {"Content-Type": "application/json"}
        status, result = SUPPORTED_TASKS[task](json)
        resp = make_response(result, status)
        resp.headers = headers
        return resp
    else:
        return 'Content-Type not supported'
    
@app.route('/memory')
@cross_origin()
def print_memory():
    return {'memory': process.memory_info().rss}    

@app.route("/snapshot")
@cross_origin()
def snap():
    global s
    if not s:
        s = tracemalloc.take_snapshot()
        return "taken snapshot\n"
    else:
        lines = []
        top_stats = tracemalloc.take_snapshot().compare_to(s, 'lineno')
        for stat in top_stats[:5]:
            lines.append(str(stat))
        return "\n".join(lines)

if __name__ == '__main__':
    app.run()