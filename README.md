# Artist Classification Model

# Project Overview

This project is a classification model that identifies the artist of a given art image. The user inputs an art image, and the website returns the top three likely artists out of the following 18:
Vincent van Gogh
Edgar Degas
Pablo Picasso
Pierre-Auguste Renoir
Albrecht Dürer
Paul Gauguin
Francisco Goya
Rembrandt
Alfred Sisley
Titian
Marc Chagall
Rene Magritte
Amedeo Modigliani
Paul Klee
Henri Matisse
Andy Warhol
Mikhail Vrubel
Sandro Botticelli
Additionally, a brief GPT-generated paragraph is provided for each of the top three predicted artists.

# Dataset

Source: [Dataset Link](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time/data)
Description: This dataset is a collection of artist names with their respective artworks.

# Pipeline Steps

<b>Data Preprocessing and Augmentation</b>

The images are preprocessed and augmented to improve the model's performance. This involves:

<li> Rescaling the image pixel values. </li>
<li> Applying random rotations, shifts, shears, and zooms.</li>
<li> Horizontally flipping some images.</li>
<li> Dividing the dataset into training and validation sets.</li>

<b> Model Architecture </b>
We use the ResNet50 model pretrained on ImageNet as the base model. The following steps are taken:

<li> Load the ResNet50 model without the top classification layer.</li>
<li> Set all layers of ResNet50 to be trainable.</li>
<li> Add new layers for our specific classification task:</li>
<li> Flatten the output of ResNet50.</li>
<li> Add a dense layer with ReLU activation and batch normalization.</li>
<li> Add another dense layer with ReLU activation and batch normalization.</li>
<li> Add the final output layer with a softmax activation function to predict probabilities for each artist class.</li>

<b> Model Training </b>
The model is trained using the augmented data, with the training process monitored through validation data. Key points include:

<li> Using Adam optimizer with a specified learning rate.</li>
<li> Compiling the model with categorical cross-entropy loss and accuracy as the metric.</li>
<li> Training the model with the training data and validating with the validation data.</li>
<li> Using callbacks to reduce the learning rate when a metric has stopped improving.</li>

# Setup Instructions

<li> Clone the repository: </li>
git clone <repository-url>
cd <repository-directory>

<li> Create and activate a virtual environment: </li>
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`

<li> Install the required packages: </li>
pip install -r requirements.txt

# Usage

After setting up the virtual environment, run the application: <b> python app.py </b>

# Contribution Guidelines

You are welcome to contribute to this project! You can fork the repository or send a message to the creator to get an invite to collaborate directly. Current needs include improving the UI and optimizing model performance.

# License

Include your license information here (e.g., MIT License).
