from flask import Flask, render_template, request, redirect
import os
import time
import temp
from tensorflow.keras.models import load_model
from imutils import paths
import imutils
import random
import cv2
from twilio.rest import Client
import yagmail

#imports for AI algorithm

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


app.config["IMAGE_UPLOADS"] = "/Users/samarth/Downloads/Dentestimate2/static/result_img"

# Global email_array
name_array = []
email_array = []
phone_array = []


@app.route('/send', methods=['GET', 'POST'])
def send():

    if request.method == "POST":

        req = request.form

        name = req.get("name")
        name_array.append(name)
        name_array[0] = name
    

        email = req.get("email")
        email_array.append(email)
        email_array[0] = email


        phone_number = req.get("phoneNumber")
        phone_array.append(phone_number)
        phone_array[0] = phone_number



        
        return redirect(request.url)



    return render_template('index.html')


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":

        if request.files:

            image = request.files["file_image"]

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], "" + email_array[0] + ".png"))

            print(image)

            np.set_printoptions(suppress=True)
            model = tensorflow.keras.models.load_model('keras_model.h5')
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            Image.open("/Users/samarth/Downloads/Dentestimate2/static/result_img/" + email_array[0] + ".png").save("/Users/samarth/Downloads/Dentestimate2/static/result_img/" + email_array[0] + ".bmp")

            image = Image.open("/Users/samarth/Downloads/Dentestimate2/static/result_img/" + email_array[0] + ".bmp")


            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            image_array = np.asarray(image)
            # image.show()

            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

            data[0] = normalized_image_array

            prediction = model.predict(data)
            
            prediction = prediction.ravel()
            prediction = prediction.tolist()
            max_value = max(prediction)

            max_index = prediction.index(max_value)

            print(prediction)
            print(max_value)
            print(max_index)

            if(max_index==0):
                return render_template('fillings.html')

            if(max_index==1):
                return render_template("crowns.html")

            if(max_index==2):
                return render_template("extractions.html")

            if(max_index==3):
                return render_template("implants.html")
            
            if(max_index==4):
                return render_template("braces.html")


            return redirect(request.url)



    return render_template("index.html")



if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get('PORT', 5000))
    app.run()
