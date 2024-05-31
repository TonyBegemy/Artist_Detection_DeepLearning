from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import cv2
import imageio
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import subprocess
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import imageio
from werkzeug.utils import secure_filename
_ = load_dotenv(find_dotenv())

def ConnectToAzure():
    """
    desc:
        Function connects to langchain AzureOpenAI
    return: 
        llm_model of llm
    """
    ## Keys ##
    OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")

    llm_model = AzureChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        azure_deployment=DEPLOYMENT_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type=OPENAI_API_TYPE,
    )
    return llm_model

def ConversationInput():
    _DEFAULT_TEMPLATE = """
    You are an expert in art and artists.
    The human has provided an image of an art piece, 
    which has been analyzed and classified with percentages indicating the likelihood of it being by specific artists. 
    You will receive these percentages and required to comment on the results.
    
    Current conversation:
    New human question: {input}
    Response:"""

    prompt = PromptTemplate(
        input_variables=["input"], template=_DEFAULT_TEMPLATE
    )

    conversation = LLMChain(
    llm=ConnectToAzure(),
    prompt=prompt,
    verbose=False,
    )
    return conversation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}# dol ely hay3ado bs 

model = load_model('model/Artist_Resnet_model_0.84.h5')


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

train_input_shape = (224, 224, 3) #3lshan el model mayfar23sh 

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

            print("\n\n\n---------------------")
            conversation = ConversationInput()
            gpt_response = conversation.predict(input = str(results))

            print(gpt_response)
            print("---------------------\n\n\n")

            return jsonify({"results": results, "image_path": f"uploads/{filename}", "gpt_response": gpt_response})
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
    app.run(host='0.0.0.0', port=5000)