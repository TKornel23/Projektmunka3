from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import flask
import io
import tensorflow as tf
from tensorflow import keras
import sys
from keras.layers import GaussianNoise
from tensorflow.python.ops.numpy_ops import np_config

app = flask.Flask(__name__)
model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = tf.keras.models.load_model('model.h5')
    print("* Model loaded")


def prepare_image(image, target, std_dev=0.1):
    # resize the input image and preprocess it
    np_config.enable_numpy_behavior()
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    # add Gaussian noise
    noise_layer = GaussianNoise(std_dev)
    image = noise_layer(image, training=True)
    
    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    print("request comes")
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            
            classes = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            
            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(299, 299)).astype('float32')/255

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict(image)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for index in range(len(classes)):
                r = {"label": classes[index], "probability": float(preds[0][index])}
                data["predictions"].append(r)

            
            # # indicate that the request was a success
            data["success"] = True
            data["result"] = classes[np.argmax(preds)]

    # return the data dictionary as a JSON response
    print(data)
    return flask.jsonify(data)
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
