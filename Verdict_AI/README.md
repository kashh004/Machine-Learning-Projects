# Indian Penal Code (IPC) Sections Predictor

## Overview

This project is an implementation of a neural network for the identification of suitable IPC sections based on an offense as input. 

## Project Structure

- **IPC.ipynb**: A Jupyter Notebook containing the project code.

## Requirements

- Python
- NumPy
- scikit-learn
- Kaggle dataset

## Data Preparation

1. Download the Indian Penal Code (IPC) Sections Information dataset from Kaggle.
2. Run the notebook step by step for data preprocessing, model training, prediction, and evaluation.

## Data Preprocessing

The data preprocessing steps in the notebook include:

1. Download the Indian Penal Code (IPC) Sections Information dataset from Kaggle.
2. Splitting the dataset into training, validation, and test sets.

## Model

The neural network model includes the following layers:

1. Input layer
2. Hidden layers with dropout for regularization
3. Output layer with softmax activation for classification


## Model

The model predicts three aspects based on the offense text:

1. **Description**: The legal description associated with the offense.
2. **Punishment**: The prescribed punishment for the offense.
3. **IPC Section**: The specific IPC section under which the offense falls.

The model uses a logistic regression classifier with TF-IDF vectorization for text feature extraction to train separate classifiers for each target (description, punishment, and IPC section).

## Usage

1. **Run the notebook**:
   - Open **IPC.ipynb** in Jupyter.
   - Follow the steps in the notebook for training the model and making predictions.

2. **Predict IPC Section**:
   - Provide an offense as input, and the model will predict the most suitable IPC section, corresponding description, and punishment.
   - Example input: `"public servant"`.

## Example

After running the notebook and training the models, use the following code snippet to predict the details:

```python
# Example usage
# Get the initial input from the user
offense_input = input("Enter the offense for prediction: ")  Ex:theft,murder etc.

# Make the prediction using the model
predicted_description = predict_description(offense_input)

# Display the prediction
print(f"Predicted IPC_Section: {predicted_description}")
