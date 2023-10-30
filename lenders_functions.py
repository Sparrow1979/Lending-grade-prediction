import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD
# from typing import List
from pandas import DataFrame


def evaluate_log_reg(data, target_variable, test_size=0.2, random_state=42):

    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logistic_model = LogisticRegression(
        class_weight='balanced', random_state=random_state)
    model = logistic_model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy:.2f}')
    print('\nConfusion Matrix:')
    print(confusion)
    print('\nClassification Report:')
    print(classification_rep)


def evaluate_models(data: pd.DataFrame, target_variable: str) -> pd.DataFrame:

    label_encoder = LabelEncoder()
    data['encoded'] = label_encoder.fit_transform(data[target_variable])

    X = data.drop([target_variable, 'encoded'], axis=1)
    y = data['encoded']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = [
        ('Decision Tree', DecisionTreeClassifier(random_state=0)),
        ('Random Forest', RandomForestClassifier(random_state=0)),
        ('SVM', SVC(random_state=0)),
        ('XGBoost', XGBClassifier(seed=0))
    ]

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for name, model in models:
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=69)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        model_names.append(name)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1_scores
    })
    return results_df


def reduced_evaluate_models(data: pd.DataFrame, target_variable: str, n_components: int):

    label_encoder = LabelEncoder()
    data['encoded'] = label_encoder.fit_transform(data[target_variable])

    X = data.drop([target_variable, 'encoded'], axis=1)
    y = data['encoded']

    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    models = [
        ('Decision Tree', DecisionTreeClassifier(random_state=0)),
        ('Random Forest', RandomForestClassifier(random_state=0)),
        ('SVM', SVC(random_state=0)),
        ('XGBoost', XGBClassifier(seed=0))
    ]

    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=69)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    svd = TruncatedSVD(n_components=n_components)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)

    for name, model in models:
        clf = Pipeline(steps=[('model', model)])

        clf.fit(X_train_reduced, y_train)

        y_pred = clf.predict(X_test_reduced)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        model_names.append(name)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    results_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1_scores
    })
    return results_df
