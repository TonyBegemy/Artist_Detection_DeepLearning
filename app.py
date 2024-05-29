from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import pickle

import imageio

app = Flask(__name__)

# Load the model
model = load_model('artist_prediction_model.h5')
labels = [...]  # Your labels list

def preprocess_image(img):
    img = cv2.resize(img, dsize=(train_input_shape[0], train_input_shape[1]))
    img = image.img_to_array(img)
    img /= 255.
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_path = "uploads/" + file.filename
            file.save(img_path)
            img = imageio.imread(img_path)
            preprocessed_img = preprocess_image(img)
            prediction = model.predict(preprocessed_img)
            prediction_probability = np.amax(prediction)
            prediction_idx = np.argmax(prediction)
            result = {
                "predicted_artist": labels[prediction_idx].replace('_', ' '),
                "prediction_probability": prediction_probability * 100
            }
            return jsonify(result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
