# USAGE
# python predict_fire.py

# import the necessary packages
from tensorflow.keras.models import load_model
from pyimagesearch import config
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os
from twilio.rest import Client

# load the trained model from disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# grab the paths to the fire and non-fire images, respectively
print("[INFO] predicting...")
firePaths = list(paths.list_images(config.FIRE_PATH))
nonFirePaths = list(paths.list_images(config.SAMPLE_FIRE_PATH))

# setup SMS alert system

# account_sid = os.environ["TWILIO_ACCOUNT_SID"]
# auth_token = os.environ["TWILIO_AUTH_TOKEN"]

# client = Client(account_sid, auth_token)

# media = "/Users/samarth/Downloads/keras-fire-detection/ForestFireOpenCV/sampdata/1010_nws_ocr-l-ahfire-052-1.jpg"

# combine the two image path lists, randomly shuffle them, and sample
# them
imagePaths = nonFirePaths
random.shuffle(imagePaths)
imagePaths = imagePaths[:config.SAMPLE_SIZE]

# loop over the sampled image paths
for (i, imagePath) in enumerate(imagePaths):
	# load the image and clone it
	image = cv2.imread(imagePath)
	output = image.copy()

	# resize the input image to be a fixed 128x128 pixels, ignoring
	# aspect ratio
	image = cv2.resize(image, (128, 128))
	image = image.astype("float32") / 255.0
		
	# make predictions on the image
	preds = model.predict(np.expand_dims(image, axis=0))[0]
	j = np.argmax(preds)
	label = config.CLASSES[j]

	# draw the activity on the output frame
	# text = label if label == "Non-Fire" else "Fire"

	if label == "Non-Fire":
		text = "Non-Fire"

	else:
		text = "Fire"

				
		# client.messages.create(
		# 	to="+19255773624",
		# 	from_="+12057720202",
		# 	body="Fire found!"
		# 	# media_url="file:///Users/samarth/Downloads/keras-fire-detection/ForestFireOpenCV/output/examples/0.png"

		# )


	output = imutils.resize(output, width=500)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX,
		1.25, (0, 255, 0), 5)

	# write the output image to disk	 
	filename = "{}.png".format(i)
	# filename = "" + output + ".png".format(i)
	p = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
	cv2.imwrite(p, output)