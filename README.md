# Multi-Class Classification with CIFAR Dataset
This repository contains a Python notebook that demonstrates multi-class classifcation using transfer learning on the CIFAR dataset. The goal is to create a model that can accurately classify images into one of the ten classes present in the CIFAR dataset.

## Dataset
The CIFAR dataset is a popular benchmark dataset for image classification tasks. It consists of 50,000 training images and 10,000 testing images, each of size 32x32 pixels. The dataset is divided into ten classes, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The notebook uses the CIFAR-10 variant, which focuses on these ten classes.

<img src="git_images/cifar.png" style="float: left; text-align: center;">

## Dependencies
To run the notebook, you need the following dependencies:
* TensorFlow
* Numpy
* Pandas
* Matplotlib
* Seaborn
* sklearn
* mlxtend
* cv2
* glob
* os

## File structure
* cifar_multi_class.ipynb:  The main Jupyter Notebook containing the code and instructions for multi-class classification on the CIFAR dataset.

* model deployment.ipynb: The Jupyter Notebook containing the code for testing and deploying the model.

* train/: Directory containing training data.
* test/: Directory containing test data.

