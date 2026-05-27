
# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting heart disease using Machine Learning algorithms. The objective is to analyze patient medical data and build predictive models capable of identifying whether a patient is likely to suffer from heart disease.

The project demonstrates the complete Machine Learning workflow including:

* Data preprocessing
* Feature engineering
* Handling imbalanced datasets
* Model training
* Hyperparameter tuning
* Model evaluation
* Visualization of results

This project was implemented using Python and various machine learning libraries from the Scikit-learn ecosystem.

---

# 📂 Dataset Description

The dataset contains medical records of patients with different health-related attributes.

## Features Used

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Resting ECG
* Maximum Heart Rate
* Exercise-Induced Angina
* ST Depression
* ST Slope

## Target Variable

* `0` → No Heart Disease
* `1` → Heart Disease Present

---

# 🛠 Technologies and Libraries Used

## Programming Language

* Python

## Libraries

* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Mlxtend

---

# 🔄 Project Workflow

## 1. Importing Libraries

Essential Python libraries were imported for:

* Data manipulation
* Data preprocessing
* Model building
* Model evaluation
* Data visualization

```python
import numpy as np
import pandas as pd
```

---

## 2. Loading the Dataset

The dataset was loaded using Pandas.

```python
data = pd.read_csv('/content/heart.csv')
```

This step converts the CSV file into a DataFrame for analysis and preprocessing.

---

## 3. Data Understanding and Exploration

Basic dataset analysis was performed using:

```python
data.dtypes
data.isnull().sum()
```

This helped identify:

* Data types of columns
* Missing values
* Structure of the dataset

---

# ⚙️ Feature and Target Separation

The dataset was divided into:

* Input Features (`x`)
* Target Variable (`y`)

```python
x = data.iloc[:,0:11]
y = data['HeartDisease']
```

---

# 🔄 Data Preprocessing

Different preprocessing techniques were applied to different column types.

## Column Categorization

```python
nomi_col = [2,6,10]
ordi_col = [1,8]
num_col = [0,3,4,5,7,9]
```

* Nominal Columns → OneHot Encoding
* Ordinal Columns → Ordinal Encoding
* Numerical Columns → Standard Scaling

---

## Column Transformer

```python
trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), nomi_col),
    (OrdinalEncoder(), ordi_col),
    (StandardScaler(), num_col),
    remainder='passthrough'
)
```

### Purpose of Preprocessing

* Convert categorical values into numerical format
* Standardize numerical values
* Improve model performance
* Prepare the dataset for machine learning algorithms

---

# ⚖️ Handling Imbalanced Dataset

The dataset was imbalanced, so Random Under Sampling was applied.

```python
from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler()
u_x, u_y = under.fit_resample(x, y)
```

### Benefits

* Balances target classes
* Reduces bias toward majority class
* Improves classification accuracy

---

# 🤖 Machine Learning Models Used

Multiple machine learning algorithms were implemented and compared.

## Models Implemented

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* Perceptron
* Bagging Classifier
* Voting Classifier

---

# 🔍 Hyperparameter Tuning

Hyperparameter optimization techniques were applied using:

```python
GridSearchCV
RandomizedSearchCV
```

### Purpose

* Improve model accuracy
* Find optimal model parameters
* Reduce overfitting

---

# 📈 Model Evaluation

The models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Metrics

## Confusion Matrix

```python
ConfusionMatrixDisplay.from_estimator(pipe4, x_test, y_test)
```

The confusion matrix helps evaluate:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

# 📉 Decision Boundary Visualization

Decision regions were visualized using Mlxtend.

```python
plot_decision_regions(
    X=x2.values,
    y=y2.values,
    clf=prc,
    legend=2
)
```

This visualization helps understand how the classifier separates different classes.

---

# 🎯 Project Outcome

The project successfully predicts heart disease using Machine Learning techniques.

The implemented system:

* Processes medical data efficiently
* Handles imbalanced datasets
* Trains multiple classification models
* Evaluates model performance effectively
* Provides prediction capability for heart disease detection

---

# 🚀 Future Enhancements

Future improvements that can be added:

* Deep Learning Models
* Real-Time Prediction System
* Flask/Streamlit Deployment
* Interactive Dashboard
* Cloud Deployment
* Improved Feature Engineering

---

# 📌 Conclusion

This project demonstrates the practical application of Machine Learning in the healthcare domain. By analyzing patient health records, the developed models can effectively predict the likelihood of heart disease.

The project highlights the importance of:

* Data preprocessing
* Feature engineering
* Model selection
* Handling imbalanced datasets
* Performance evaluation

This system can assist healthcare professionals in early diagnosis and decision-making processes.

---

# 👨‍💻 Author

**Gowtham**

Artificial Intelligence & Data Science Student
