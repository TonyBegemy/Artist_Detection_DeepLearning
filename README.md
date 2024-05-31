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

<li> Working with 18 Classes only (artists that have above than 150 paintings) </li>
<li> Initializing class weights to each artist </li>
<li> Rescaling the image pixel values to [0,1]</li>
<li> Applying random rotations, shifts, shears, and zooms (Augmentation).</li>
<li> Dividing the dataset into training and validation sets (80%, 20%).</li>

<b> Model Architecture </b>
We use the ResNet50 model pretrained on ImageNet as the base model. The following steps are taken:

<li> Loaded the ResNet50 model without the top classification layer.</li>
<li> Added new layers for our specific classification task:</li>
<li> Flattened the output of ResNet50.</li>
<li> Added a dense layer with 512 units, ReLU activation, dropout, and batch normalization.</li>
<li> Added another dense layer with 32 units, ReLU activation, dropout, and batch normalization.</li>
<li> Added the final output layer with a softmax activation function to predict probabilities for each artist class.</li>

<b> Model Training </b>
The model is trained using the augmented data, with the training process monitored through validation data. Key points include:

<li> Using Adam optimizer with a specified learning rate.</li>
<li> Compiling the model with categorical cross-entropy loss and accuracy as the metric.</li>
<li> Using callbacks EarlyStopping and ReduceLROnPlateau to reduce the learning rate when a metric (val_loss) has stopped improving.</li>
<li> First Training Phase: Training the entire network for 20 epochs.</li>
<li> Second Training Phase: Freeze the core ResNet layers and train only the first 50 layers for 50 epochs.</li>

# Setup Instructions

<li> Clone the repository: </li>
git clone <repository-url>
cd <repository-directory>

<li> Create and activate a virtual environment: </li>
python -m venv venv
# On Mac use: source venv/bin/activate   # On Windows use: venv\Scripts\activate

<li> Install the required packages: </li>
pip install -r requirements.txt

# Usage

Run the model.py file to get the model saved in a model path
Run the application: <b> python app.py </b>
Use the test_images folder to test our model :)

P.S: if the application doesn't have an "uploads" folder, make sure to create that.

# Contribution Guidelines

You are welcome to contribute to this project! You can fork the repository or send a message to the creator to get an invite to collaborate directly. Current needs include improving the UI and optimizing model performance.

# License

MIT License
