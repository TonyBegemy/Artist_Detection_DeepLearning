from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import cv2
import imageio
import os
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}# dol ely hay3ado bs 

model = load_model('model/Artist_Resnet_model.h5')


# Define the labels (artists) used in your model
labels = [
    "Vincent_van_Gogh",
    "Edgar_Degas",
    "Pablo_Picasso",
    "Pierre-Auguste_Renoir",
    "Albrecht_Dürer",
    "Paul_Gauguin",
    "Francisco_Goya",
    "Rembrandt",
    "Alfred_Sisley",
    "Titian",
    "Marc_Chagall",
    "Rene_Magritte",
    "Amedeo_Modigliani",
    "Paul_Klee",
    "Henri_Matisse",
    "Andy_Warhol",
    "Mikhail_Vrubel",
    "Sandro_Botticelli"
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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            img = imageio.imread(file_path)
            preprocessed_img = preprocess_image(img)
            prediction = model.predict(preprocessed_img)
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            results = [
                {"artist": labels[idx].replace('_', ' '), "probability": prediction[0][idx] * 100}
                for idx in top_3_indices
            ]
            return jsonify({"results": results, "image_path": f"uploads/{filename}"})
        else:
            return jsonify({'error': 'File type not allowed'})
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)