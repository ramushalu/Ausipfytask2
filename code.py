# TASK 2 - NETFLIX CONTENT TYPE PREDICTION MODEL
# Goal:
# Predict whether a Netflix title is a Movie or TV Show.

# Workflow:
# 1. Load and explore dataset
# 2. Select relevant features
# 3. Handle missing values
# 4. Encode categorical variables
# 5. Split data into training and testing sets
# 6. Train classification models
# 7. Evaluate model performance
# 8. Compare model accuracy
# 9. Test a custom prediction
# 10. Save results and trained model

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

print("Libraries imported successfully!")

# 2. LOAD DATASET

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
display(df.head())

# 3. BASIC DATA EXPLORATION

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

# 4. TARGET VARIABLE ANALYSIS

print("\nContent Type Distribution:")
print(df["type"].value_counts())

plt.figure(figsize=(7, 5))

df["type"].value_counts().plot(kind="bar")

plt.title("Netflix Content Type Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")
plt.xticks(rotation=0)

plt.show()

# 5. SELECT FEATURES
# We do not use title or show_id because they are identifiers,
# not useful predictive features.

features = [
    "director",
    "country",
    "release_year",
    "rating",
    "duration",
    "listed_in"
]

X = df[features].copy()
y = df["type"].copy()

print("\nSelected Features:")
for feature in features:
    print("-", feature)

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

# 6. DEFINE FEATURE TYPES

categorical_features = [
    "director",
    "country",
    "rating",
    "duration",
    "listed_in"
]

numerical_features = [
    "release_year"
]

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 8. PREPROCESSING PIPELINE
# Categorical data -> One-Hot Encoding
# Numerical data -> Missing values handled using median

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_transformer,
            categorical_features
        ),
        (
            "numerical",
            numerical_transformer,
            numerical_features
        )
    ]
)

print("\nPreprocessing pipeline created successfully!")

# 9. MODEL 1 - LOGISTIC REGRESSION

print("\n" + "=" * 60)
print("MODEL 1 - LOGISTIC REGRESSION")
print("=" * 60)

logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

y_pred_lr = logistic_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
)

print("\nLogistic Regression Accuracy:")
print(f"{lr_accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_lr
    )
)


# Confusion Matrix

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr,
    labels=["Movie", "TV Show"]
)

disp_lr = ConfusionMatrixDisplay(
    confusion_matrix=cm_lr,
    display_labels=["Movie", "TV Show"]
)

disp_lr.plot()

plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# 10. MODEL 2 - DECISION TREE

print("\n" + "=" * 60)
print("MODEL 2 - DECISION TREE")
print("=" * 60)

decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            )
        )
    ]
)

decision_tree_model.fit(
    X_train,
    y_train
)

y_pred_dt = decision_tree_model.predict(
    X_test
)

dt_accuracy = accuracy_score(
    y_test,
    y_pred_dt
)

print("\nDecision Tree Accuracy:")
print(f"{dt_accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_dt
    )
)


# Confusion Matrix

cm_dt = confusion_matrix(
    y_test,
    y_pred_dt,
    labels=["Movie", "TV Show"]
)

disp_dt = ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Movie", "TV Show"]
)

disp_dt.plot()

plt.title("Decision Tree - Confusion Matrix")
plt.show()

# 11. MODEL 3 - RANDOM FOREST

print("\n" + "=" * 60)
print("MODEL 3 - RANDOM FOREST")
print("=" * 60)

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_rf = random_forest_model.predict(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("\nRandom Forest Accuracy:")
print(f"{rf_accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_rf
    )
)


# Confusion Matrix

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf,
    labels=["Movie", "TV Show"]
)

disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf,
    display_labels=["Movie", "TV Show"]
)

disp_rf.plot()

plt.title("Random Forest - Confusion Matrix")
plt.show()

# 12. COMPARE ALL MODELS

print("\n" + "=" * 60)
print("MODEL ACCURACY COMPARISON")
print("=" * 60)

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        dt_accuracy,
        rf_accuracy
    ]
})

results = results.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

display(results)

# 13. ACCURACY COMPARISON GRAPH

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Netflix Content Type Prediction - Model Accuracy")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.xticks(rotation=15)

for i, value in enumerate(results["Accuracy"]):
    plt.text(
        i,
        value + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.show()

# 14. CUSTOM PREDICTION FUNCTION

def predict_content_type(
    director,
    country,
    release_year,
    rating,
    duration,
    listed_in
):

    new_data = pd.DataFrame({
        "director": [director],
        "country": [country],
        "release_year": [release_year],
        "rating": [rating],
        "duration": [duration],
        "listed_in": [listed_in]
    })

    prediction = random_forest_model.predict(
        new_data
    )

    return prediction[0]

# 15. TEST CUSTOM PREDICTION

print("\n" + "=" * 60)
print("CUSTOM CONTENT TYPE PREDICTION")
print("=" * 60)

prediction = predict_content_type(
    director="Not Given",
    country="United States",
    release_year=2020,
    rating="TV-MA",
    duration="90 min",
    listed_in="Dramas, Comedies"
)

print("\nInput:")
print("Director: Not Given")
print("Country: United States")
print("Release Year: 2020")
print("Rating: TV-MA")
print("Duration: 90 min")
print("Genre: Dramas, Comedies")

print("\nPredicted Content Type:")
print(prediction)

# 16. SAVE MODEL RESULTS

results.to_csv(
    "content_type_model_results.csv",
    index=False
)

print("\nModel comparison results saved as:")
print("content_type_model_results.csv")

# 17. SAVE TRAINED RANDOM FOREST MODEL

joblib.dump(
    random_forest_model,
    "netflix_content_type_model.pkl"
)

print("\nTrained model saved as:")
print("netflix_content_type_model.pkl")

# 18. FINAL PROJECT SUMMARY

print("\n")
print("=" * 60)
print("TASK 2 - PROJECT SUMMARY")
print("=" * 60)

print("\nDataset:")
print(f"Total records: {len(df)}")

print("\nTarget Variable:")
print("Movie vs TV Show")

print("\nNumber of Features Used:")
print(len(features))

print("\nFeatures Used:")
for feature in features:
    print("-", feature)

print("\nModel Performance:")

for _, row in results.iterrows():

    print(
        f"{row['Model']}: "
        f"{row['Accuracy']:.4f}"
    )

best_model_name = results.iloc[0]["Model"]
best_accuracy = results.iloc[0]["Accuracy"]

print("\nBest Test Accuracy:")
print(best_model_name)

print(f"Accuracy: {best_accuracy:.4f}")

print("\nFiles Created:")
print("- content_type_model_results.csv")
print("- netflix_content_type_model.pkl")

print("\n" + "=" * 60)
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("=" * 60)

output:

Libraries imported successfully!

Dataset loaded successfully!
Dataset shape: (8790, 10)

First 5 rows:
show_id	type	title	director	country	date_added	release_year	rating	duration	listed_in
0	s1	Movie	Dick Johnson Is Dead	Kirsten Johnson	United States	9/25/2021	2020	PG-13	90 min	Documentaries
1	s3	TV Show	Ganglands	Julien Leclercq	France	9/24/2021	2021	TV-MA	1 Season	Crime TV Shows, International TV Shows, TV Act...
2	s6	TV Show	Midnight Mass	Mike Flanagan	United States	9/24/2021	2021	TV-MA	1 Season	TV Dramas, TV Horror, TV Mysteries
3	s14	Movie	Confessions of an Invisible Girl	Bruno Garotti	Brazil	9/22/2021	2021	TV-PG	91 min	Children & Family Movies, Comedies
4	s8	Movie	Sankofa	Haile Gerima	United States	9/24/2021	1993	TV-MA	125 min	Dramas, Independent Movies, International Movies


Column names:
['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']

Dataset information:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 8790 entries, 0 to 8789
Data columns (total 10 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       8790 non-null   object
 1   type          8790 non-null   object
 2   title         8790 non-null   object
 3   director      8790 non-null   object
 4   country       8790 non-null   object
 5   date_added    8790 non-null   object
 6   release_year  8790 non-null   int64 
 7   rating        8790 non-null   object
 8   duration      8790 non-null   object
 9   listed_in     8790 non-null   object
dtypes: int64(1), object(9)
memory usage: 686.8+ KB

Missing values:
show_id         0
type            0
title           0
director        0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
dtype: int64

Duplicate rows:
0

Content Type Distribution:
type
Movie      6126
TV Show    2664
Name: count, dtype: int64


Selected Features:
- director
- country
- release_year
- rating
- duration
- listed_in

Feature shape: (8790, 6)
Target shape: (8790,)

Categorical Features:
['director', 'country', 'rating', 'duration', 'listed_in']

Numerical Features:
['release_year']

Training samples: 7032
Testing samples: 1758

Preprocessing pipeline created successfully!

============================================================
MODEL 1 - LOGISTIC REGRESSION
============================================================

Logistic Regression Accuracy:
0.9954

Classification Report:
              precision    recall  f1-score   support

       Movie       0.99      1.00      1.00      1225
     TV Show       1.00      0.98      0.99       533

    accuracy                           1.00      1758
   macro avg       1.00      0.99      0.99      1758
weighted avg       1.00      1.00      1.00      1758



============================================================
MODEL 2 - DECISION TREE
============================================================

Decision Tree Accuracy:
0.9903

Classification Report:
              precision    recall  f1-score   support

       Movie       1.00      0.99      0.99      1225
     TV Show       0.97      1.00      0.98       533

    accuracy                           0.99      1758
   macro avg       0.99      0.99      0.99      1758
weighted avg       0.99      0.99      0.99      1758



============================================================
MODEL 3 - RANDOM FOREST
============================================================

Random Forest Accuracy:
0.9596

Classification Report:
              precision    recall  f1-score   support

       Movie       0.95      1.00      0.97      1225
     TV Show       1.00      0.87      0.93       533

    accuracy                           0.96      1758
   macro avg       0.97      0.93      0.95      1758
weighted avg       0.96      0.96      0.96      1758



============================================================
MODEL ACCURACY COMPARISON
============================================================
Model	Accuracy
0	Logistic Regression	0.995449
1	Decision Tree	0.990330
2	Random Forest	0.959613



============================================================
CUSTOM CONTENT TYPE PREDICTION
============================================================

Input:
Director: Not Given
Country: United States
Release Year: 2020
Rating: TV-MA
Duration: 90 min
Genre: Dramas, Comedies

Predicted Content Type:
Movie

Model comparison results saved as:
content_type_model_results.csv

Trained model saved as:
netflix_content_type_model.pkl


============================================================
TASK 2 - PROJECT SUMMARY
============================================================

Dataset:
Total records: 8790

Target Variable:
Movie vs TV Show

Number of Features Used:
6

Features Used:
- director
- country
- release_year
- rating
- duration
- listed_in

Model Performance:
Logistic Regression: 0.9954
Decision Tree: 0.9903
Random Forest: 0.9596

Best Test Accuracy:
Logistic Regression
Accuracy: 0.9954

Files Created:
- content_type_model_results.csv
- netflix_content_type_model.pkl

============================================================
TASK 2 COMPLETED SUCCESSFULLY!
============================================================
