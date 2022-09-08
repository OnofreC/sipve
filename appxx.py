"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image  #Permite la edición de imágenes directo con python

import torch
from flask import Flask, render_template, request, redirect

app = Flask(__name__)

RESULT_FOLDER = os.path.join("static")
app.config["RESULT_FOLDER"] = RESULT_FOLDER
imgs = [img]  # batched list of images

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        #results = model(img, size=640)
        results = model(img, size=640)

        # for debugging
        # data = results.pandas().xyxy[0].to_json(orient="records")
        # return data
        # results = img_bytes

        results.save(save_dir="static")

        full_filename = os.path.join(app.config["RESULT_FOLDER"], 'results0.jpg')
        return redirect("static/image0.jpg")
    return render_template("index.html")
"""
        results.render()  # Se actualiza la imagen con las etiquetas de clases objetos encontrados.
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save("static/image0.jpg", format="JPEG")
        return redirect("static/image0.jpg")

    return render_template("index.html")"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    
    #model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo repositorio local para cargar todo el local
    model = torch.hub.load(
        'ultralytics/yolov5', 'custom', path='50best1508.pt', force_reload=True, autoshape=True
       
        #"ultralytics/yolov5", "yolov5", pretrained=True, force_reload=True, autoshape=True
       
    )  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
