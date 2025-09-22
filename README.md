A Keras-based CNN for Marine Debris
This repository contains a self-contained test script for a deep learning model designed to classify marine debris using image data. The script uses a Convolutional Neural Network (CNN) architecture built with Keras to categorize debris into 10 different classes.

🎯 Project Objective
The primary objective of this script is to automatically detect and classify waste types using images from a water tank environment. The model is trained to recognize the following categories:

bottle

can

chain

drink-carton

hook

propeller

shampoo-bottle

standing-bottle

tire

valve

🛠️ Setup and Execution
To run the script on your machine, you'll need the following Python libraries.

pip install tensorflow
pip install keras
pip install numpy
pip install pandas
pip install scikit-learn
pip install opencv-python
pip install tqdm

📂 Script Structure
The provided code is a single script designed to perform a complete test run from data loading to saving results. The main functions are:

mu(): The primary function that orchestrates the entire training and testing process for a specific run.

TrainModel (class): Handles the model's definition and data preparation.

getDatas(): Loads and preprocesses the image dataset from the file system.

Test3(): Evaluates the trained model on the dataset and generates a confusion matrix.

modelsave(), modelhistsave(), savetext(): Functions to save the trained model, its history, and the final classification report.

Example Usage
The mu function is configured to run with specific parameters. You can modify the function call within the script to change the test run's configuration. For example, the following command trains and tests the model with 64x64 pixel, 3-channel (color) images and a batch size of 200:

mu(size=64, channel=3, batchsize=200, filtre=False)

🤖 Model Architecture
The model is a simple CNN architecture built using the Keras library.

Convolutional2D and MaxPooling2D layers are used to extract features from the images.

Dropout layers are added to prevent overfitting.

A Flatten layer prepares the data for the fully connected (Dense) layers.

The final layer uses a softmax activation function to output predictions for the 10 categories.

The model is compiled with categorical_crossentropy loss and the adam optimizer.

📝 Results and Saving
The script saves all results from the test run in various formats:

Model file: The trained model is saved with a *.h5 extension, containing its structure and weights.

Training history: The model's training and validation loss/accuracy for each epoch are saved in a hist*.csv file.

Confusion matrix: An output*.csv file provides a confusion matrix of the model's classification performance on the test data.

Percentage report: The output*.txt and yüzdelikler*.csv files contain the individual and overall prediction percentages for the test run.

🤝 Contributing
This script is intended as a demonstration of a test run. However, if you would like to contribute, you can submit a pull request or open an issue to suggest improvements or enhancements.
