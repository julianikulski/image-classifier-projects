# Developing an AI application to classify flowers

This project is part of Udacity's Data Science Nanodegree Term 1. The purpose of the project is to leverage deep learning algorthims to build a model that can read in pictures of flowers and return the correct flower name.

## Table of Contents
* [Installation](#Installation)
* [Project Motivation](#motivation)
* [File Description](#description)
* [Results](#Results)
* [Licensing, Authors, Acknowledgements](#licensing)

## Installation
The code requires Python versions of 3.* and general libraries available through the Anaconda package. In particular, torchvision is used as a library.

## Project Motivation <a name="motivation"></a>
This project is part of the work required by the Data Science Nanodegree Program offered by Udacity. Torchvision is being leveraged to create a deep learning model and train it.

## File Description <a name="description"></a>
There is one Jupyter notebook containing the code to train the deep learning model. Markdown cells were used to help understand the different steps taken. There is also a json file that contains a mapping between flower categories and names. Moreover, four py files are part of the project: train.py trains the model created in the Jupyter notebook and save a checkpoint; predict.py uses the trained model to classify an image; utility.py contains a transformation function; helper.py contains a function to define the network architecture and a function defining the training and validation loops.

## Results
The resulting project allows to use `python train.py data_directory` to train the model on the pictures in the data_directory and then predict the flower name with `python predict.py /path/to/image checkpoint`.

The train command has multiple options that can be specified:
* `python train.py data_dir --save_dir save_directory` to set a directory for the checkpoint
* `python train.py data_dir --arch "vgg13"` choosing a particular architecture
* `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20` setting hyperparameters
* `python train.py data_dir --gpu` to use gpu mode for training

Customized options are available for the predict function as well:
* `python predict.py input checkpoint --top_k 3` returning the top k classes
* `python predict.py input checkpoint --category_names cat_to_name.json` returning the real flower names leveraging the mapping json file
* `python predict.py input checkpoint --gpu` using gpu more for inference

## Licensing, Authors, Acknowledgements <a name="licensing"></a>
The underlying data used for the analysis comes from Udacity and is not available publicly. Most of the information in the markdown cells was provided by Udacity and is subject to their [licensing](https://eu.udacity.com/legal/terms-of-use).
