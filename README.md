# ❤️ HEART DISEASE PREDICTION USING MACHINE LEARNING

## 📌 INTRODUCTION

Heart disease is one of the leading causes of death worldwide. Early prediction and diagnosis of heart-related diseases can help healthcare professionals provide timely treatment and reduce the risk of severe medical complications. With the advancement of Artificial Intelligence and Machine Learning technologies, predictive systems can analyze medical data efficiently and support clinical decision-making.

This project, **Heart Disease Prediction using Machine Learning**, focuses on developing a predictive model capable of determining whether a patient is likely to suffer from heart disease based on several medical parameters. The project demonstrates the practical implementation of Machine Learning concepts such as data preprocessing, feature engineering, handling imbalanced datasets, model training, hyperparameter tuning, evaluation, and visualization.

The developed system can assist healthcare professionals by providing an intelligent prediction mechanism for early heart disease detection.

---

## 🎯 PROJECT OBJECTIVE

The main objective of this project is to build a Machine Learning model that can accurately predict the presence of heart disease using patient medical data.

The project aims to:

- Analyze and understand medical datasets
- Perform data preprocessing and feature transformation
- Handle imbalanced datasets effectively
- Train multiple Machine Learning classification models
- Compare model performances
- Visualize prediction results
- Improve model accuracy using hyperparameter tuning

The system is designed to support healthcare professionals in making informed medical decisions.

---

## 📂 DATASET DESCRIPTION

The dataset used in this project contains several medical attributes collected from patients. Each row represents patient medical information along with a target variable indicating whether the patient has heart disease.

### FEATURES INCLUDED IN THE DATASET

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise-Induced Angina
- ST Depression
- ST Slope

### TARGET VARIABLE

The target column is:

- `HeartDisease`

Where:
- `0` → No Heart Disease
- `1` → Heart Disease Present

This problem is treated as a binary classification problem in Machine Learning.

---

## 🛠 TECHNOLOGIES AND LIBRARIES USED

### PROGRAMMING LANGUAGE

- Python

### PYTHON LIBRARIES

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Mlxtend

### DEVELOPMENT TOOLS

- Jupyter Notebook
- Anaconda

---

## 🔄 PROJECT WORKFLOW

The project follows a complete Machine Learning pipeline from data loading to prediction and evaluation.

---

## 1️⃣ IMPORTING LIBRARIES

The first step of the project involves importing all necessary Python libraries required for data manipulation, preprocessing, machine learning model development, and visualization.

### LIBRARIES USED

- NumPy → Numerical operations and array handling
- Pandas → Data manipulation and analysis
- Scikit-learn → Machine learning algorithms and preprocessing
- Matplotlib → Data visualization
- Imbalanced-learn → Handling imbalanced datasets
- Mlxtend → Decision boundary visualization

These libraries provide efficient tools for implementing Machine Learning workflows.

```python
import numpy as np
import pandas as pd
```

---

## 2️⃣ LOADING THE DATASET

The dataset is loaded using the Pandas library.

```python
data = pd.read_csv('/content/heart.csv')
```

The CSV file is converted into a DataFrame, which allows easy analysis and manipulation of data.

After loading the dataset, initial exploration is performed to understand:
- Number of rows and columns
- Data types
- Missing values
- Unique values

---

## 3️⃣ DATA EXPLORATION AND UNDERSTANDING

Data exploration is an important step used to understand the structure and quality of the dataset.

### OPERATIONS PERFORMED

- Checking data types
- Finding missing values
- Viewing unique values
- Understanding feature distributions

```python
data.dtypes
data.isnull().sum()
```

This step helps determine:
- Which columns are categorical
- Which columns are numerical
- Whether data cleaning is required

---

## 4️⃣ FEATURE AND TARGET SEPARATION

The dataset is divided into:
- Input Features (`x`)
- Target Variable (`y`)

```python
x = data.iloc[:,0:11]
y = data['HeartDisease']
```

### PURPOSE

- Features are used to train the Machine Learning model
- The target variable represents the prediction output

---

## 5️⃣ DATA PREPROCESSING

Machine Learning algorithms cannot directly process categorical data efficiently. Therefore, preprocessing techniques are applied to transform the data into suitable numerical formats.

### COLUMN CATEGORIZATION

```python
nomi_col=[2,6,10]
ordi_col=[1,8]
num_col=[0,3,4,5,7,9]
```

- Nominal categorical columns
- Ordinal categorical columns
- Numerical columns

---

## 6️⃣ FEATURE TRANSFORMATION USING COLUMN TRANSFORMER

Different preprocessing techniques are applied to different column types using `make_column_transformer()`.

### PREPROCESSING TECHNIQUES USED

#### ONE HOT ENCODER
Used for nominal categorical columns.

#### ORDINAL ENCODER
Used for ordinal categorical columns.

#### STANDARD SCALER
Used for numerical columns.

### COLUMN TRANSFORMER

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import make_column_transformer

trans = make_column_transformer(
    (OneHotEncoder(sparse_output=False), nomi_col),
    (OrdinalEncoder(), ordi_col),
    (StandardScaler(), num_col),
    remainder='passthrough'
)
```

This preprocessing pipeline prepares the dataset for Machine Learning algorithms.

---

## ⚖️ 7️⃣ HANDLING IMBALANCED DATASET

Medical datasets often contain imbalanced target classes where one class has significantly more samples than the other.

To solve this issue, Random Under Sampling was applied.

```python
from imblearn.under_sampling import RandomUnderSampler

under = RandomUnderSampler()
u_x,u_y = under.fit_resample(x,y)
```

### BENEFITS OF UNDER SAMPLING

- Balances target class distribution
- Reduces model bias toward majority class
- Improves classification performance

---

## 🤖 8️⃣ MACHINE LEARNING MODELS USED

Multiple Machine Learning classification algorithms were implemented and evaluated.

### MODELS IMPLEMENTED

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- Perceptron
- Bagging Classifier
- Voting Classifier

---

## 🔍 9️⃣ HYPERPARAMETER TUNING

Hyperparameter optimization techniques were applied to improve model performance.

### TECHNIQUES USED

- GridSearchCV
- RandomizedSearchCV

### BENEFITS

- Improves model accuracy
- Reduces overfitting
- Finds optimal model parameters

---

## 📈 🔟 MODEL EVALUATION

After training, the models were evaluated using various evaluation metrics.

### EVALUATION METRICS USED

- Accuracy Score
- Confusion Matrix
- Classification Metrics

### CONFUSION MATRIX

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(pipe4, x_test, y_test)
```

The confusion matrix helps measure:
- True Positives
- True Negatives
- False Positives
- False Negatives

---

## 📉 1️⃣1️⃣ DECISION BOUNDARY VISUALIZATION

Decision regions were visualized using the Mlxtend library.

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(
    X=x2.values,
    y=y2.values,
    clf=prc,
    legend=2
)
```

### PURPOSE

- Visualize classifier decision boundaries
- Understand class separation
- Analyze model behavior visually

---

## 🎯 PROJECT OUTCOME

The project successfully predicts heart disease using Machine Learning techniques by:

- Processing patient medical data
- Performing feature transformation and preprocessing
- Handling imbalanced datasets
- Training multiple classification models
- Evaluating prediction accuracy

The developed system demonstrates how Machine Learning can support healthcare professionals in early disease detection and diagnosis.

---

## 🚀 FUTURE ENHANCEMENTS

Future improvements for this project may include:

- Deep Learning Models
- Real-Time Prediction Systems
- Flask or Streamlit Deployment
- Interactive Dashboards
- Cloud Deployment
- Explainable AI Techniques
- Larger Medical Datasets

---

## 📌 CONCLUSION

This project demonstrates the practical implementation of Machine Learning in the healthcare sector. By analyzing patient medical records and applying classification algorithms, the system can effectively predict the likelihood of heart disease.

The project highlights the importance of:
- Data preprocessing
- Feature engineering
- Model selection
- Hyperparameter tuning
- Handling imbalanced datasets
- Performance evaluation

Overall, the developed system provides a valuable example of how Artificial Intelligence and Data Science can contribute to healthcare solutions and assist medical professionals in making informed decisions.

---
### 📂 Dataset
https://github.com/islavathgowtham/heart-disease-prediction-ML/blob/main/heart.csv
### 💻 Source Code
https://github.com/gowtham/heart-disease-prediction-ml

---

# 👨‍💻 AUTHOR

## GOWTHAM ISLAVATH

Artificial Intelligence & Data Science Student
