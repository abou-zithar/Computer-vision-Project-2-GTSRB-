# German Traffic Sign Recognition with Deep Learning

This project demonstrates the implementation of a deep learning model for the task of image classification on the German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is deployed using Streamlit to provide a user-friendly web interface.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Deployment](#deployment)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Introduction

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is a popular dataset for training image classification models to recognize various traffic signs. This project involves building a convolutional neural network (CNN) to classify traffic signs and deploying the model using Streamlit.

## Dataset

The GTSRB dataset contains images of traffic signs in different conditions. Each image is labeled with the type of traffic sign it represents. The dataset is publicly available and can be downloaded from [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) designed to effectively recognize and classify traffic signs. The architecture consists of the following layers:

- Input layer
- Convolutional layers with ReLU activation
- Max pooling layers
- Fully connected (dense) layers
- Output layer with softmax activation

## Training

The model is trained using the GTSRB dataset. Key steps in the training process include:

1. Data Preprocessing:
   - Resizing images
   - Normalizing pixel values
   - One-hot encoding labels

2. Data Augmentation:
   - Random rotations
   - Shifts
   - Flips

3. Compilation:
   - Optimizer: Adam
   - Loss function: Categorical Cross-Entropy
   - Metrics: Accuracy

4. Training:
   - Training on training set
   - Validation on validation set

## Deployment

Streamlit is used to deploy the trained model. The deployment process involves:

1. Creating a Streamlit app (`app.py`).
2. Loading the trained model.
3. Setting up file upload functionality to allow users to upload images.
4. Displaying the prediction results on the web interface.

## Results

The final model achieves high accuracy on the validation set. Below is an example of the output from the Streamlit app:

![Model Output]
<a href="https://imagetolink.com/ib/nQfZU3vlCZ"><img src="https://imagetolink.com/ib/nQfZU3vlCZ.png" alt="nQfZU3vlCZ"/></a>

## Usage

To use the Streamlit app, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/gtsrb-image-classification.git
   cd gtsrb-image-classification


 ##  Run the Streamlit app:
 ```sh
   streamlit run app.py
```
## Conclusion
This project demonstrates a complete workflow for building and deploying a
deep learning model for image classification on the GTSRB dataset. The use of
 Streamlit makes it easy to interact with the model through a web interface.
