# House Price Prediction Using Boston Dataset and Regression Model

Welcome to the House Price Prediction project! This repository contains code and resources for building a regression model to predict house prices using the Boston Housing dataset.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project aims to predict house prices in Boston using various regression models. The Boston Housing dataset, which is a well-known dataset in the machine learning community, provides various features of houses and their corresponding prices. By exploring and analyzing this data, we can build a model that accurately predicts house prices based on the given features.

## Dataset

The dataset used in this project is the Boston Housing dataset, which can be accessed through the `sklearn` library. The dataset consists of 506 samples and 14 features, including:

- CRIM: per capita crime rate by town
- ZN: proportion of residential land zoned for lots over 25,000 sq. ft.
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- NOX: nitrogen oxides concentration (parts per 10 million)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centers
- RAD: index of accessibility to radial highways
- TAX: full-value property tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- LSTAT: percentage of lower status of the population
- MEDV: median value of owner-occupied homes in $1000s (target variable)

## Project Structure

The project repository is structured as follows:

```
house-price-prediction/
├── data/
│   └── boston.csv          # Dataset (if applicable)
├── notebooks/
│   └── EDA.ipynb           # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py  # Data preprocessing scripts
│   ├── model.py               # Model training scripts
│   └── evaluate.py            # Model evaluation scripts
├── tests/
│   └── test_model.py          # Unit tests for model
├── README.md
├── requirements.txt           # Python package dependencies
└── .gitignore
```

## Installation

To run this project, you need to have Python installed. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Exploratory Data Analysis (EDA):**
   - Navigate to the `notebooks/` directory and open `EDA.ipynb` to explore the dataset and visualize various features.

2. **Data Preprocessing:**
   - Run the data preprocessing script to clean and prepare the data for modeling:
     ```bash
     python src/data_preprocessing.py
     ```

3. **Model Training:**
   - Train the regression model using the prepared dataset:
     ```bash
     python src/model.py
     ```

4. **Model Evaluation:**
   - Evaluate the model's performance on the test set:
     ```bash
     python src/evaluate.py
     ```

## Model

In this project, we experiment with various regression models including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regression
- Random Forest Regression

The final model's performance is evaluated based on metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

## Results

The results of the model evaluation, including performance metrics and visualizations, are documented in the `notebooks/` directory. Here, we compare the performance of different models and select the best one based on their predictive accuracy.

## Contributing

Contributions are welcome! If you have any improvements or suggestions, please create a pull request or open an issue.
