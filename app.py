from flask import Flask, request, render_template, jsonify, send_from_directory
import numpy as np
import cv2
import imageio
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from werkzeug.utils import secure_filename

_ = load_dotenv(find_dotenv())

def ConnectToAzure():
    """
    Connect to Azure's OpenAI service and return a configured model instance.

    Returns:
             llm_model: an instance of AzureChatOpenAI initialized with environment variables.
    """
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
    """
    Create a conversational model using Azure's OpenAI service with a specific prompt template.

    Returns:
             conversation: an instance of LLMChain initialized with a prompt template and the Azure model.
    """
    _DEFAULT_TEMPLATE = """
    You are an expert in art and artists.
    The human has provided an image of an art piece, 
    which has been analyzed and classified with percentages indicating the likelihood of it being by specific artists. 
    You will receive these percentages and required to comment on the results.
    and tell why they are very similar
    
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

ALLOWED_EXTENSIONS = {'jpg','jpeg'}

model = load_model('model/Artist_Resnet_model_0.84.h5')

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

train_input_shape = (224, 224, 3)

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.

    Parameters:
             filename(str): the name of the file to check.

    Returns:
             bool: True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    """
    Preprocess the input image for the model.

    Parameters:
             img: the image to preprocess.

    Returns:
             img: the preprocessed image ready for prediction.
    """
    img = cv2.resize(img, (train_input_shape[0], train_input_shape[1]))
    img = image.img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def about():
    """
    Render the about page.

    Returns:
             HTML: the rendered about.html page.
    """
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handle image upload and prediction requests. If the request is POST, process the uploaded file, 
    make a prediction, and get a GPT response. If the request is GET, render the upload form.

    Returns:
             JSON: the prediction results and GPT response if POST, or the rendered index.html page if GET.
    """
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

            conversation = ConversationInput()
            gpt_response = conversation.predict(input=str(results))

            return jsonify({"results": results, "image_path": f"uploads/{filename}", "gpt_response": gpt_response})
        else:
            return jsonify({'error': 'File type not allowed'})
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve the uploaded file from the upload directory.

    Parameters:
             filename(str): the name of the file to serve.

    Returns:
             File: the requested file from the upload directory.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
