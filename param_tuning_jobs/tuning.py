# --- Import packages

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# --- Scikit Learn Packages
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import metrics
from sklearn.model_selection import (
    TimeSeriesSplit,
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC

import graphviz

# --- Other Packages
import pathlib
import sys
import os
import argparse
import logging
import pickle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--filename",
    help="Name of data file without extension",
    type=str,
    required=True,
    dest="filename",
)
args = parser.parse_args()

# --- Read data

file_path: pathlib.Path = os.path.join(
    os.path.abspath("./data"), args.filename + ".csv"
)


# categorical feature list
class FeatureSpace:
    cat_features = ["school_nlns"]  # categorical features
    true_false_levels = ["f", "t"]  # levels for cat_features
    num_features = [
        "great_messages_proportion",
        "teacher_referred_count",
        "non_teacher_referred_count",
        "students_reached",
    ]  # numeric feature list
    target = ["is_exciting"]  # Target variagle


def read_data(file_path: pathlib.Path):
    """
    info: Reads csv file into memory
    input: file_path
    output: dataframe
    """

    try:
        df_ = pd.read_csv(
            filepath_or_buffer=file_path,
            infer_datetime_format=True,
            encoding="ISO-8859-1",
        )
    except Exception as error:
        print(error)
    return df_


def data_prep():
    """
    info: takes a data frame and does train test plit using timesplit
    input: no input
    output: tuple of dataframes (X_train, y_train, X_test, y_test)
    """
    df_projects = read_data(file_path=file_path)

    # ---- sort the data by date_posted for  time splitting
    df_projects_sorted = df_projects.sort_values(by="date_posted").copy()

    # ---- generate the split indexes.
    tscv = TimeSeriesSplit(n_splits=2)
    _, split_two = tscv.split(df_projects_sorted)

    # ---- Generate split train_test split indexes for based on the entire dataset
    train_index, test_index = split_two

    print("number of observations in training: {}".format(train_index.shape[0]))
    print("number of observations in training: {}\n".format(test_index.shape[0]))

    print(
        "train pct: {:2.0%}".format(train_index.shape[0] / df_projects_sorted.shape[0])
    )
    print(
        "test  pct: {:2.0%}".format(test_index.shape[0] / df_projects_sorted.shape[0])
    )
    # ---- generate the split indexes to use to sample the data
    _, sample_split = tscv.split(df_projects_sorted.iloc[train_index, :])

    # ---- Generate split sample train_test split indexes
    sample_train_idx, sample_test_idx = sample_split

    print(
        "number of sample observations for training: {}".format(
            sample_train_idx.shape[0]
        )
    )
    print(
        "number of sample observations for testing: {}\n".format(
            sample_test_idx.shape[0]
        )
    )

    print(
        "sample train pct: {:2.0%}".format(
            sample_train_idx.shape[0] / df_projects_sorted.iloc[train_index, :].shape[0]
        )
    )
    print(
        "sample test  pct: {:2.0%}".format(
            sample_test_idx.shape[0] / df_projects_sorted.iloc[train_index, :].shape[0]
        )
    )

    X = df_projects_sorted[FeatureSpace.cat_features + FeatureSpace.num_features]
    y = (
        df_projects_sorted[FeatureSpace.target]
        .is_exciting.map({"t": 1, "f": 0})
        .to_frame()
    )

    # Training and test set for the full data
    X_train = X.iloc[train_index, :]
    y_train = y.iloc[train_index, :]

    X_test = X.iloc[test_index, :]
    y_test = y.iloc[test_index, :]

    # Training and test set for the sample data
    X_train_sample = X.iloc[sample_train_idx, :]
    y_train_sample = y.iloc[sample_train_idx, :]

    X_test_sample = X.iloc[sample_test_idx, :]
    y_test_sample = y.iloc[sample_test_idx, :]
    return X_train, y_train, X_test, y_test


def pipeline_ops():
    # categorical pipeline
    categorical_pipe = Pipeline(
        [
            (
                "label",
                OrdinalEncoder(
                    categories=[FeatureSpace.true_false_levels],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            )  # -- note this is from category-encoders not kslearn
        ]
    )

    # numeric pipeline
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    # Data processing pipeline

    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, FeatureSpace.cat_features),
            ("num", numerical_pipe, FeatureSpace.num_features),
        ]
    )
    return preprocessing


# Evaluation metrics function
def eval_metrics(model, X, y_true):
    auc = metrics.roc_auc_score(y_true=y_true, y_score=model.predict(X))
    accuracy = metrics.accuracy_score(y_true=y_true, y_pred=model.predict(X))
    recall = metrics.recall_score(y_true=y_true, y_pred=model.predict(X))
    precision = metrics.precision_score(y_true=y_true, y_pred=model.predict(X))
    log_loss = metrics.log_loss(y_true=y_true, y_pred=model.predict(X))
    return {
        "roc_auc": round(auc, 4),
        "min_log_loss": round(log_loss, 4),
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 2),
        "precision": round(precision, 2),
    }


def model_pipe_log(X_train, y_train):
    # Pipeline for model
    estimator = Pipeline(
        [
            ("preprocess", pipeline_ops()),
            ("regressor", LogisticRegression(verbose=1, max_iter=500)),
        ]
    )

    # model fitting
    np.random.seed(123)
    estimator = estimator.fit(X_train, y_train.values.ravel())
    return estimator


def model_pipe_rf(X_train, y_train):
    # ---- Param Grid for Randomized search CV
    random_grid = {
        "n_estimators": [500, 1000, 1500],
        "max_depth": [3, 5, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    tsplit = TimeSeriesSplit(n_splits=5)

    # ---- Randomized search cv
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=random_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=tsplit,
        random_state=42,
        verbose=1,
    )

    # ---- Pipeline for Random Forest tuning
    rf_pipe = Pipeline([("preprocess", pipeline_ops()), ("regressor", random_search)])

    # ---- Model fitting
    rf = rf_pipe.fit(X_train, y_train.values.ravel())
    return rf


def model_pipe_dtc(X_train, y_train):
    # ---- Param Grid for Randomized search CV
    param_grid = {
        "splitter": ["best", "random"],
        "max_depth": [1, 3, 5, 7, 9, 11, 12],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "max_features": ["auto", "log2", "sqrt", None],
        "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    }

    tsplit = TimeSeriesSplit(n_splits=5)

    # ---- Randomized search cv
    random_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(),
        param_distributions=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=tsplit,
        verbose=1,
    )

    # ---- Pipeline for Random Forest tuning
    dtc_pipe = Pipeline(
        [
            ("preprocess", pipeline_ops()),
            ("scaler", StandardScaler()),
            ("regressor", random_search),
        ]
    )

    # ---- Model fitting
    dtc = dtc_pipe.fit(X_train, y_train.values.ravel())
    return dtc


def model_pipe_svm(X_train, y_train):
    # ---- Param Grid for Randomized search CV
    param_grid = {
        "C": np.logspace(-1, 1, 3),  # Regularization parameter.
        "kernel": ["rbf", "poly"],  # Kernel type
        "gamma": np.logspace(-1, 1, 3).tolist()
        + [
            "scale",
            "auto",
        ],  # Gamma is the Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    }

    tsplit = TimeSeriesSplit(n_splits=5)

    # ---- Randomized search cv
    grid_search = RandomizedSearchCV(
        estimator=SVC(),
        param_distributions=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=tsplit,
        verbose=1,
    )

    # ---- Pipeline for Random Forest tuning
    svm_pipe = Pipeline(
        [
            ("preprocess", pipeline_ops()),
            ("scaler", StandardScaler()),
            ("regressor", grid_search),
        ]
    )

    # ---- Model fitting
    svm = svm_pipe.fit(X_train, y_train.values.ravel())
    return svm


def main():
    X_train, y_train, X_test, y_test = data_prep()
    estimator_log = model_pipe_log(
        X_train=X_train,
        y_train=y_train,
    )
    log_train = eval_metrics(model=estimator_log, X=X_train, y_true=y_train)
    log_test = eval_metrics(model=estimator_log, X=X_test, y_true=y_test)
    print(f"\n log_train: {log_train}")
    print(f"\n log_train: {log_test}")
    with open("./models/log.pickle", "wb") as f:
        pickle.dump(estimator_log, f)

    estimator_rf = model_pipe_rf(
        X_train=X_train,
        y_train=y_train,
    )
    rf_train = eval_metrics(model=estimator_rf, X=X_train, y_true=y_train)
    rf_test = eval_metrics(model=estimator_rf, X=X_test, y_true=y_test)
    print(f"\n rf_train: {rf_train}")
    print(f"\n rf_test: {rf_test}")
    with open("./models/rf.pickle", "wb") as f:
        pickle.dump(estimator_rf, f)

    estimator_dtc = model_pipe_dtc(
        X_train=X_train,
        y_train=y_train,
    )
    dtc_train = eval_metrics(model=estimator_dtc, X=X_train, y_true=y_train)
    dtc_test = eval_metrics(model=estimator_dtc, X=X_test, y_true=y_test)
    print(f"\n dtc_train: {dtc_train}")
    print(f"\n dtc_test: {dtc_test}")
    with open("./models/dtc.pickle", "wb") as f:
        pickle.dump(estimator_dtc, f)

    estimator_svm = model_pipe_svm(
        X_train=X_train,
        y_train=y_train,
    )
    svm_train = eval_metrics(model=estimator_svm, X=X_test, y_true=y_test)
    svm_test = eval_metrics(model=estimator_svm, X=X_test, y_true=y_test)
    print(f"\n svm_train: {svm_train}")
    print(f"\n svm_test: {svm_test}")
    with open("./models/svm.pickle", "wb") as f:
        pickle.dump(estimator_svm, f)
    metric_result = pd.DataFrame(
        [log_train, log_test, rf_train, rf_test, svm_train, svm_test],
        index=["log_train", "log_test", "rf_train", "rf_test", "svm_train", "svm_test"],
    )
    metric_result.to_csv("./results/metric_result.csv")
    return print("\n all jobs completed")


if __name__ == "__main__":
    main()
