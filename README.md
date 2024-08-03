# Iris Classification with MLflow

This project demonstrates the use of MLflow for tracking machine learning experiments. We use the Iris dataset to train a RandomForest classifier and log parameters, metrics, and the trained model to MLflow.

## Repository Structure

- **MLproject**: Defines the MLflow project.
- **iris_classification.py**: Script to train the model and log results.
- **conda.yaml**: Conda environment definition file (optional).
- **README.md**: Project documentation (optional).

## How to Run

1. Clone the repository.
2. Run the project using MLflow with the command:
   ```bash
   mlflow run .
