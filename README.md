# Facial Recognition using CNN

## Description

Facial recognition is a widely used biometric method that identifies individuals based on their unique facial features. It is used in various applications, such as flight check-in, social media tagging, and personalized advertising. In this project, we use **Convolutional Neural Networks (CNNs)** to build a facial recognition system capable of accurately recognizing faces from images. The project uses the **ORL Face Database**, a dataset consisting of 400 images, to train and test the model.

## Table of Contents

1. [Description](#description)
2. [Technology Used](#technology-used)
3. [Dependencies](#dependencies)
4. [Steps to Build the Model](#steps-to-build-the-model)
5. [Project Insights](#project-insights)

## Technology Used

- **Keras**: For building and training the Convolutional Neural Network (CNN).
- **TensorFlow**: Backend for Keras to handle deep learning computations.
- **Scikit-learn**: For splitting the dataset and evaluating model performance.
- **Matplotlib**: For visualizing the model's results during training.
- **NumPy**: For numerical operations and image handling.


## Dependencies

- `keras`
- `tensorflow`
- `scikit-learn`
- `matplotlib`
- `numpy`

## Steps to Build the Model

### 1. Input the Required Libraries

First, the necessary libraries like Keras, OpenCV, and others are imported to handle data processing, model building, and evaluation.

### 2. Load the Dataset

The **ORL Face Database**, which consists of 400 images of 40 people (10 images per person), is loaded. The images are 112x92 pixels and are taken under varying conditions such as different facial expressions and lighting.

### 3. Normalize the Images

To ensure uniformity, all images are normalized to a specific scale to help the CNN model learn better and faster.

### 4. Split the Dataset

The dataset is split into a training and testing set using an appropriate technique, ensuring that the model learns general features and doesn't overfit.

### 5. Transform the Images

The images are resized to equal dimensions to feed them into the Convolutional Neural Network (CNN). This step ensures that each image has the same shape for processing.

### 6. Build the CNN Model

The CNN model consists of three main layers:

- **Convolutional Layer**: To extract features from the images.
- **Pooling Layer**: To reduce the spatial size of the feature maps, allowing the model to focus on more important features.
- **Fully Connected Layer**: To classify the extracted features and output predictions.

### 7. Train the Model

The model is trained using the dataset. The optimizer and loss functions are selected to minimize error and enhance model performance.

### 8. Plot the Results

The results, including accuracy and loss over training epochs, are plotted to visualize the model's performance and training progress.

### 9. Iterate Until Accuracy is Above 90%

The model is iterated by adjusting hyperparameters, architectures, and training methods until an accuracy of above 90% is achieved on the test data.

## Project Insights

### Dataset Details:
- **ORL Face Database**: The dataset consists of 400 images of 40 individuals, with 10 images per person.
- The images vary in terms of lighting, facial expression, and orientation, providing a challenge for the model to recognize faces under different conditions.

### CNN Model Details:
- The model uses a **Convolutional Neural Network (CNN)**, which is well-suited for image recognition tasks. It captures spatial hierarchies in images using convolution and pooling operations, which helps improve accuracy.
- The **Fully Connected Layer** at the end of the model ensures that the extracted features are classified into one of the 40 possible classes (representing the 40 people in the dataset).

### Performance:
- The model is trained and iterated to achieve **over 90% accuracy** on facial recognition tasks. It uses techniques such as image normalization and data augmentation to ensure robustness against various image conditions (lighting, angles, etc.).

---

This **`README.md`** gives an overview of the **Facial Recognition using CNN** project, including the dataset details, steps for building the model, and technologies used. Let me know if you'd like to modify or expand any part of it!

