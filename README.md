# Weather-Classification-ML

# Weather Classification using Machine Learning

This repository contains a Python project that applies machine learning techniques to classify weather types based on various features. The project demonstrates data preprocessing, visualization, model training, and evaluation using Random Forest and Support Vector Machine (SVM) classifiers.

## Project Overview

- **Data Loading & Exploration:**  
  Reads a CSV dataset and performs an initial inspection using Pandas (`df.head()`, `df.isnull().sum()`).

- **Data Visualization:**  
  Utilizes Seaborn and Matplotlib to create pairplots, scatter plots, and a correlation heatmap for exploratory data analysis.

- **Data Preprocessing:**  
  - Encodes categorical variables using factorization.  
  - Selects features based on correlation with the target variable.  
  - Splits the data into training and testing sets and scales features using `StandardScaler`.

- **Model Training & Evaluation:**  
  - Implements a custom function to train models over several epochs.  
  - Trains both a Random Forest Classifier and an SVM (with a fixed C value).  
  - Plots training and validation accuracies, computes confusion matrices, and generates classification reports.

## Repository Structure

- `weather_classification_data.csv`  
  The dataset used for training and evaluating the models.

- `weather_classification.py`  
  The main Python script containing the data preprocessing, visualization, and model training code.

- `README.md`  
  This file.

## Prerequisites

Make sure you have Python 3 installed. The following Python packages are required:
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

pip install pandas matplotlib seaborn scikit-learn
How to Run
Clone the repository:


`git clone https://github.com/<your-username>/Weather-Classification-ML.git
cd Weather-Classification-ML`

Ensure that the dataset (weather_classification_data.csv) is in the correct path or update the path in the script accordingly.

Run the Python script:

`python weather_classification.py`

The script will:

Load and visualize the data.
Preprocess the data (including encoding categorical variables and feature scaling).
Train two machine learning models (Random Forest and SVC) and evaluate their performance.
Display accuracy plots, confusion matrices, and detailed classification reports.
Contributing
Contributions are welcome! If you have suggestions or improvements, please fork the repository and create a pull request with your changes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
This project was inspired by Kaggle competitions and various machine learning tutorials available online. Special thanks to the open-source community for providing robust libraries and resources.









