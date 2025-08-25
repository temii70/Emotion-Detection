# Emotion-Detection
Emotion Detection with CNN

This project uses a Convolutional Neural Network (CNN) built with PyTorch to detect emotions from facial images. It includes data preprocessing, model training with early stopping, evaluation, and visualization of results.

Features
Custom CNN architecture for emotion classification
Data augmentation and normalization
Early stopping to prevent overfitting
Training and test loss visualization
Webcam-based real-time emotion detection (with OpenCV)
Easy-to-follow Jupyter notebook


Getting Started
Prerequisites:
Python 3.9
See requirements.txt for all dependencies

Installation
1. Clone the repository:
    git clone https://github.com/temii70/Emotion-Detection.git
    cd Emotion-Detection

2. Install dependencies:
    pip install -r requirements.txt

3. Download and extract the dataset from https://www.kaggle.com/datasets/msambare/fer2013  into the train and test folders.

Usage
-  Run model.ipynb in Jupyter Notebook or VS Code to train and  evaluate the model.
-  For real-time emotion detection, use the OpenCV webcam script provided.

Project Structure
* model.ipynb — Main notebook for training and evaluation
* model.py — CNN model definition
* requirements.txt — List of required Python packages
* archive — Folder for training and test images (not included in repo)

Results
    Training and test loss curves are plotted in the notebook.
    Model accuracy is printed after evaluation.
    License


This project is licensed under the Apache License.






