# Image-Segmentation

This repository contains my image segmentation task.<br>
The model takes in road images and segments them into several classes like background, car, person etc.<br>
Main segment of the code is in the "UNet-Image-segmentation.ipynb" file. Run the file to train or test models.

## Model
You can access the model via https://drive.google.com/file/d/1SXTo0KCX5QmZHS7hZcDshdD95VzYFXmL/view?usp=share_link .

## UNet.py
Contains the model architecture.

## config.py
Contains the training and testing parameters like batch-size, base directory location etc.

## dataloader.py
Contains the dataloader to load data to the model to be trained. The loader creates random data batches of a fixed size.

## emptycheck.py
Checks whether directory is empty. Useful for automatically getting any empty classes or labels.

## preprocess.py
For preprocessing all the data into suitable format. Like changing image size to be fit to be trained.

## testing.py
Contains code for testing the model

## training.py
Contains code for training the model
