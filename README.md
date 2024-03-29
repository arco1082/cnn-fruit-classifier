# cnn-fruit-classifier
PyTorch CNN convolutional network to classify Fruit images


# Project

Create a machine learning algorithm that will classify 114 fruit types.  The dataset was provided by Kaggle.  First step to setup project is to create an account in Kaggle, and download this dataset and place it in the "images" directory relative to the Jupyter notebook at the root of this repository:
https://www.kaggle.com/moltean/fruits

A few steps afterward:
1. Run the notebook in a GPU enabled environment
2. Run all the sections of the notebook
3. Train the custom CNN model
4. Run the test app below, it allows to pass in images that are locally available on the project

# Libraries 

Here are some libraries used for both data processing and training
1. For training, we will use the PyTorch framewok
2. For dataset transform and utilities, we will use torch utilities to normalize our dataset and provide data augmentation
3. The last part of the notebook also shows how to run the training job in Sagemaker, so imported the sagemaker library as well
4. Numpy used for data structure processing
5. PIL for image operations

# Sagemaker

1. There is a source_pytorch directory where the scripts necessary to run training in Sagemaker is located.  It should be able to run pretty much as is as long as the images from the dataset are uploaded to an S3 bucket (its commented out, but removing the comment should upload a local "images" folder to an S3 bucket)

# About the CNN network

For this project, I used a custom CNN network with 5 convolutional layers and 2 fully connected layers.  The best loss achieved was 0.06 and for the training set provided in the Kaggle dataset we were able to get 98% accuracy.