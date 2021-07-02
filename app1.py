from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import base64
import cv2
from io import BytesIO
from PIL import Image
import h5py
import os
from flask_ngrok import run_with_ngrok

INPUT_IMAGE = "/content/input.jpg"
OUTPUT_IMAGE = "/content/output.jpg"


#app
app = Flask(__name__)
run_with_ngrok(app)

#routes
@app.route('/',methods = ['POST'])

def post():
    image = request.get_json()
    image_string = base64.b64decode(image['file'])
    image_data = BytesIO(image_string)
    img = Image.open(image_data)
    img.save(INPUT_IMAGE)
    segment_image = custom_segmentation()
    segment_image.inferConfig(num_classes= 1, class_names= "scratch")
    segment_image.load_model("mask_rcnn_model.020-1.088950.h5")
    segment_image.segmentImage(INPUT_IMAGE, output_image_name=OUTPUT_IMAGE, show_bboxes=True)
    
    with open(OUTPUT_IMAGE, "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
        final_base64_image_string = my_string.decode('utf-8')
    return {"output_image":  final_base64_image_string}
    
if __name__ == '__main__':
    app.run()
