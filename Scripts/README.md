This repository contains Python modules to handle end-to-end processes for working with Excel data, performing exploratory analysis, preprocessing, and building models for credit score calculation. It includes tools for data visualization, model building, and evaluation to help identify the best-performing model.

Features

Read Excel Data: Load and convert Excel files into pandas DataFrames for seamless data manipulation.

Exploratory Data Analysis (EDA): Perform descriptive analysis and gain insights into the data distribution.

Data Visualization: Generate visualizations to better understand patterns, relationships, and trends in the data.

Preprocessing: Clean, preprocess, and prepare the data for model training (e.g., handling missing values, encoding, scaling).

Credit Score Calculation: Implement algorithms to calculate credit scores based on the input data.

Model Building: Train and evaluate multiple machine learning models.

Model Evaluation: Assess models based on metrics and select the best-performing model.


Prerequisites

Before running the modules, ensure you have the following dependencies installed:

Python (>= 3.7)

pandas

numpy

matplotlib

seaborn

scikit-learn

File Structure

data_loader.py: Functions to load and process Excel files.

eda.py: Functions for performing exploratory data analysis.

plot.py: Functions to create visualizations.

preprocessing.py: Data preprocessing utilities.

credit_scoring.py: Credit scoring algorithms.

train_model.py: Functions to train machine learning models and evaluate and compare models.