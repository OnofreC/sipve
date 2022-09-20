import io
import os
import json
import argparse
from PIL import Image
import subprocess

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
#parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
#parser.add_argument("--port", default=5000, type=int, help="port number")
#args = parser.parse_args()

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='50best1508.pt', force_reload=True, autoshape=True)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='static/50best1508.pt', force_reload=True).autoshape()
model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results
@app.route("/opencam", methods=['GET'])
def opencam():
    print("here")
    subprocess.run(['python', 'detect.py', '--weights','50best1508.pt', '--source', '0'],shell=True)
    return "done"
    

#@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return

        img_bytes = file.read()
        results = get_prediction(img_bytes)

        results.save(save_dir='static')

        full_filename = os.path.join(app.config['RESULT_FOLDER'], 'results0.jpg')
        return redirect('static/image0.jpg')
    return render_template('index.html')
app.run()
#app.run(host="0.0.0.0", port=args.port)      