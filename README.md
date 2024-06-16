# Cat and Dog Image Classification using MobileNetV2

This project uses a pretrained MobileNetV2 model to classify images of cats and dogs. The dataset consists of images collected from the "Cat" and "Dog" categories.

## Overview

The project involves several key steps:

1. **Data Collection and Exploration:**
   - Images are collected from the directory "D:/Deep Learning/Projects/Datasets/Cat_Dog".
   - Initial exploration is done to determine the number of images and their distribution between cats and dogs.

2. **Data Preprocessing:**
   - Images are resized to a uniform size of 224x224 pixels using PIL (Python Imaging Library).
   - Images are converted to numpy arrays and normalized to [0, 1] range.
   - Labels are assigned based on file names (0 for cats and 1 for dogs).
   - The dataset is split into training and testing sets using `train_test_split` from `sklearn.model_selection`.

3. **Building the Model:**
   - The MobileNetV2 model pretrained on ImageNet is loaded without the top layer.
   - Additional layers are added for global average pooling and a dense layer with softmax activation for classification.
   - The model is compiled with Adam optimizer and SparseCategoricalCrossentropy loss function.

4. **Training and Evaluation:**
   - The model is trained on the training set for 5 epochs.
   - Model performance is evaluated using the test set to determine accuracy.

5. **Prediction:**
   - An interface is provided to input a path to a new image to be classified.
   - The image is resized, normalized, and reshaped to match the input shape expected by the model.
   - The model predicts whether the image represents a cat or a dog and outputs the result.

## Files Included

- `README.md`: This file providing an overview of the project.
- `cat_dog_classification.ipynb`: Jupyter notebook containing the complete code for data preprocessing, model building, training, evaluation, and prediction.
- `Cat_Dog`: Folder containing the original dataset of images classified as cats and dogs.
- `Cat_Dog_resized`: Folder containing the resized images of cats and dogs, standardized to 224x224 pixels.

## Libraries Used

- `numpy` for numerical computations.
- `PIL` (Python Imaging Library) for image resizing and manipulation.
- `matplotlib` for data visualization.
- `sklearn` for dataset splitting and evaluation metrics.
- `tensorflow` and `keras` for building and training the neural network model.

## Usage

To run the project:
1. Ensure you have Python installed along with the necessary libraries (`numpy`, `PIL`, `matplotlib`, `sklearn`, `tensorflow`, `keras`).
2. Execute the code in `cat_dog_classification.ipynb` in a Jupyter notebook environment or any Python IDE.

## Conclusion

This project demonstrates the application of transfer learning using a pretrained MobileNetV2 model for image classification. By leveraging a pretrained model, the project achieves efficient training and good accuracy in distinguishing between images of cats and dogs.

