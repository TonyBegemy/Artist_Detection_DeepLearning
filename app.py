from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
import imageio
import os
import pickle
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}# dol ely hay3ado bs 

# Load the trained model from the .pkl file
with open('artist_prediction_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the labels (artists) used in your model
labels = [
#    placeholders leh asamy el artists  # Add all the labels here
]


train_input_shape = (224, 224, 3)#3lshan el model mayfar23sh 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = cv2.resize(img, (train_input_shape[0], train_input_shape[1]))
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
        else:
            return jsonify({'error': 'File type not allowed'})
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
