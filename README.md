# Ausipfytask2

# Netflix Content Type Prediction Model

## 📌 Project Overview

This project is part of the **Machine Learning Internship Program**.

The objective of this task is to develop a machine learning model that predicts whether a Netflix title is a **Movie** or a **TV Show** based on the available content features.

The project follows a complete machine learning workflow including data preparation, feature selection, categorical encoding, model training, evaluation, and model comparison.

---

## 🎯 Objective

The main objective is to build a **binary classification model** that predicts the content type:

* **Movie**
* **TV Show**

The model uses information such as director, country, release year, rating, duration, and genre/category information.

---

## 📋 Task Workflow

The project follows the workflow given in the internship task:

1. Select relevant dataset features.
2. Encode categorical variables.
3. Train classification models.
4. Evaluate prediction performance.
5. Compare model accuracy.

---

## 📂 Dataset

The Netflix dataset contains information about Netflix movies and TV shows.

### Main Features Used

| Feature        | Description                                |
| -------------- | ------------------------------------------ |
| `director`     | Director of the title                      |
| `country`      | Country associated with the title          |
| `release_year` | Year the title was released                |
| `rating`       | Content rating                             |
| `duration`     | Duration of the movie or number of seasons |
| `listed_in`    | Genre/category information                 |
| `type`         | Target variable: Movie or TV Show          |

The target variable is:

```text
type
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Google Colab / Jupyter Notebook

---

## 🔧 Data Preprocessing

The following preprocessing steps were performed:

* Selected relevant features.
* Separated input features and target variable.
* Handled missing values using imputation.
* Encoded categorical variables using **One-Hot Encoding**.
* Used median imputation for the numerical feature.
* Split the dataset into training and testing sets.
* Used stratified splitting to maintain the distribution of Movie and TV Show classes.

---

## 🤖 Machine Learning Models

Three classification algorithms were trained and compared:

### 1. Logistic Regression

A linear classification algorithm used as a baseline model.

### 2. Decision Tree Classifier

A tree-based classification algorithm that makes predictions using decision rules.

### 3. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

---

## 📊 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

A comparison chart was also generated to compare the accuracy of the trained models.

---

## 🔍 Prediction

After training the models, a sample Netflix title was provided to demonstrate how the trained model predicts whether the content is a **Movie** or **TV Show**.

---

## 📁 Project Files

```text
Task-2-Content-Type-Prediction/
│
├── Dataset.csv
├── content_type_prediction.ipynb
├── content_type_model_results.csv
├── netflix_content_type_model.pkl
├── screenshots/
└── README.md
```

### File Description

* `Dataset.csv` – Netflix dataset used for the project.
* `content_type_prediction.ipynb` – Complete machine learning implementation.
* `content_type_model_results.csv` – Model performance comparison results.
* `netflix_content_type_model.pkl` – Saved trained machine learning model.
* `screenshots/` – Screenshots of important outputs and results.
* `README.md` – Project documentation.

---

## 📈 Results

The performance of Logistic Regression, Decision Tree, and Random Forest models was compared using classification metrics.

The model comparison helps identify how different classification algorithms perform on the Netflix content type prediction problem.

The detailed results are available in:

```text
content_type_model_results.csv
```

---

## 💡 Key Learning Outcomes

Through this project, I learned:

* Binary classification
* Feature selection
* Categorical data encoding
* Data preprocessing
* Train-test splitting
* Logistic Regression
* Decision Tree Classification
* Random Forest Classification
* Model evaluation
* Confusion matrix analysis
* Model comparison
* Saving trained ML models

---

## ✅ Conclusion

The Netflix Content Type Prediction project demonstrates a complete machine learning classification workflow.

The trained models use Netflix content attributes to predict whether a title belongs to the **Movie** or **TV Show** category. Different classification algorithms were compared using standard evaluation metrics.

This project helped develop practical skills in **classification algorithms, data encoding, model validation, and machine learning workflow development**.

---

## 👩‍💻 Internship Project

**Machine Learning Internship Program**

**Task:** Task 2 – Netflix Content Type Prediction Model

**Domain:** Machine Learning

**Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Google Colab
