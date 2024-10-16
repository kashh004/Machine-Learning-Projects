# Automated Model Optimization for Breast Cancer Diagnosis using TPOT

This project leverages **TPOT**, a genetic programming-based AutoML tool, to optimize machine learning models for breast cancer diagnosis. The TPOT framework automatically selects, tunes, and evaluates models, ensuring the best possible model for predicting whether a tumor is malignant or benign.

## Project Overview

The key components of this project include:

1. **Breast Cancer Dataset**: The dataset is used for binary classification to diagnose breast cancer as malignant or benign.
2. **AutoML with TPOT**: Automatically finds and tunes machine learning models using TPOT.
3. **Model Evaluation**: After optimization, the best model is selected, evaluated, and exported for future use.

## How It Works

- **TPOT AutoML**: TPOT automates the model selection and hyperparameter tuning process. By fitting the model to the training data, it explores different pipelines and finds the most effective one.
- **Dataset**: The project uses the **Breast Cancer Wisconsin Dataset** from scikit-learn, which consists of 30 features and a binary target variable.

## Steps

1. **Install TPOT**:
   ```bash
   pip install tpot
2.	**Exported Pipeline**:
The best_model_pipeline.py file contains the pipeline with the best model and hyperparameters discovered by TPOT

## Requirements
  •	**Python 3.6+**
	•	**TPOT**
	•	**scikit-learn**
	•	**pandas**
## Results
  - **Best model accuracy: 95%**
  -	**The optimized model pipeline is stored in best_model_pipeline.py**

