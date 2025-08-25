# Emotion-Detection
Emotion Detection with CNN

This project uses a Convolutional Neural Network (CNN) built with PyTorch to detect emotions from facial images.
The model is trained on the Fer 2013 dataset (32,000+) labelled images.
 It includes data preprocessing, model training with early stopping, evaluation, and visualization of results.

Features:
- Custom CNN architecture for emotion classification
- Data augmentation and normalization
- Early stopping to prevent overfitting
- Training and test loss visualization
- Webcam-based real-time emotion detection (with OpenCV)
- Easy-to-follow Jupyter notebook


# Getting Started
Prerequisites>Python 3.9 & Cuda


See requirements.txt for all dependencies

Installation
1. Clone the repository:
    git clone https://github.com/temii70/Emotion-Detection.git
    
2. Install dependencies:
    pip install -r requirements.txt

3. Download and extract the dataset from https://www.kaggle.com/datasets/msambare/fer2013  into the train and test folders.

Usage
-  Run model.ipynb in Jupyter Notebook or your IDE to train and  evaluate the model.
-  For real-time emotion detection, run the emotion_script.ipynb  provided.

Project Structure
* model.ipynb — Main notebook for training and evaluation
* model.py — CNN model definition
* requirements.txt — List of required Python packages
* archive — Folder for training and test images (not included in repo)


This project is licensed under the Apache License.


# Demo:
Webcam Live Detection:
![Live Demo](demos/readme.gif)

Works on multiple faces at the same time:
![Live Demo](demos/readme2.gif)


Next Steps:
    This is a ongoing project, and as such I will be updating/improving the model and adding other applications as time progresses.