from flask import Flask, request, make_response
from impulse_response import run_ir_task
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_impulse_response_task(request_json):
    if "payload" not in request_json:
        return False, "Request Body is missing a 'payload' entry"
    if "sample-rate" not in request_json:
        return False, "Request Body us missing a 'sample-rate' entry"
    if "P" not in request_json:
        return False, "Request Body us missing a 'P' entry"
    
    recordedSignals = request_json["payload"]
    sampleRate = request_json["sample-rate"]
    P = request_json["P"]
    result = run_ir_task(recordedSignals, P, sampleRate)
    
    return True, {
        "inverted-impulse-response": result
    }

def run_volume_task(request_json):
    return ''

SUPPORTED_TASKS = {
    'impulse-response': run_impulse_response_task,
    'volume': run_volume_task
}

@app.route("/task/<string:task>", methods=['POST'])
def task_handler(task):
    if task not in SUPPORTED_TASKS:
        return 'ERROR'
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        headers = {"Content-Type": "application/json"}
        status, result = SUPPORTED_TASKS[task](json)
        resp = make_response(result, status)
        resp.headers = headers
        return resp
    else:
        return 'Content-Type not supported'
    
if __name__ == '__main__':
    app.run()