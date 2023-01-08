# --- Import packages

import pandas as pd
import numpy as np


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

# --- Other Packages
import pathlib
import sys
import os
import argparse
import logging
import pickle

# --- Define loggers
logging.basicConfig(filename='tune.log', filemode='w'  format='%(asctime)s - %(message)s', level=logging.INFO)


# --- Defined filename as script input
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


# categorical feature container
class FeatureSpace:
    cat_features = [
        "teacher_teach_for_america",
        "one_non_teacher_referred_donor_g",
        "eligible_double_your_impact_matc",
    ]  # categorical features
    true_false_levels = ["f", "t"]  # levels for cat_features
    metro_levels = ["rural", "suburban", "urban"]
    teacher_levels = ["Ms.", "Mrs.", "Mr.", "Dr."]
    resource_type_levels = [
        "Visitors",
        "Trips",
        "Other",
        "Books",
        "Technology",
        "Supplies",
    ]
    poverty_levels = [
        "low poverty",
        "moderate poverty",
        "high poverty",
        "highest poverty",
    ]
    primary_area_levels = [
        "Health & Sports",
        "History & Civics",
        "Applied Learning",
        "Special Needs",
        "Math & Science",
        "Literacy & Language",
    ]

    num_features = [
        "great_messages_proportion",
        "teacher_referred_count",
        "non_teacher_referred_count",
        "students_reached",
        "fulfillment_labor_materials",
        "total_price_excluding_optional_s",
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
        logging.error(error)
    return df_


def data_prep():
    """
    info: takes a data frame and does train test plit using timesplit
    input: no input
    output: tuple of dataframes (X_train, y_train, X_test, y_test)
    """
    # --- read data
    df_projects = read_data(file_path=file_path)

    # ---- sort the data by date_posted for  time splitting
    df_projects_sorted = df_projects.sort_values(by="date_posted").copy()

    # ---- generate the split indexes.
    tscv = TimeSeriesSplit(n_splits=2)
    _, split_two = tscv.split(df_projects_sorted)

    # ---- Generate split train_test split indexes for based on the entire dataset
    train_index, test_index = split_two

    logging.info("number of observations in training: {}".format(train_index.shape[0]))
    logging.info("number of observations in training: {}\n".format(test_index.shape[0]))

    logging.info(
        "train pct: {:2.0%}".format(train_index.shape[0] / df_projects_sorted.shape[0])
    )
    logging.info(
        "test  pct: {:2.0%}".format(test_index.shape[0] / df_projects_sorted.shape[0])
    )
    # ---- generate the split indexes to use to sample the data
    _, sample_split = tscv.split(df_projects_sorted.iloc[train_index, :])

    # ---- Generate split sample train_test split indexes
    sample_train_idx, sample_test_idx = sample_split

    logging.info(
        "number of sample observations for training: {}".format(
            sample_train_idx.shape[0]
        )
    )
    logging.info(
        "number of sample observations for testing: {}\n".format(
            sample_test_idx.shape[0]
        )
    )

    logging.info(
        "sample train pct: {:2.0%}".format(
            sample_train_idx.shape[0] / df_projects_sorted.iloc[train_index, :].shape[0]
        )
    )
    logging.info(
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
    """
    info: defined pipelines for data preprocessing
    input: none
    output: data preprocessing pipe
    """
    # --- categorical pipeline
    categorical_pipe = Pipeline(
        [
            (
                "label",
                OrdinalEncoder(
                    categories=[
                        FeatureSpace.true_false_levels,
                        FeatureSpace.true_false_levels,
                        FeatureSpace.true_false_levels,
                    ],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-2,
                ),
            )  # -- note this is from category-encoders not kslearn
        ]
    )

    # --- numeric pipeline
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    # --- Data processing pipeline
    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, FeatureSpace.cat_features),
            ("num", numerical_pipe, FeatureSpace.num_features),
        ]
    )
    return preprocessing


# Evaluation metrics function
def eval_metrics(model, X, y_true):
    """
    info: function to evaluate metrics
    input: model of type sklean predictor,X = input features,y = response labels
    output: dictionary of metrics
    """
    # --- various evaluation metrics
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
    """
    info: pipe for logistic regresion
    input: input features, response variable
    output: fitted logistic pipe
    """
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
    """
    info: model pipe for random forest
    input: input features, response labels
    output: fitted predictor pipe for random forest
    """
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
        scoring="accuracy",
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
    """
    info: define model pipe for decision tree
    input: n x n array input features, response label
    output: fitted predictor pipe for decision tree
    """
    # ---- Param Grid for Randomized search CV
    param_grid = {
        "splitter": ["best", "random"],
        "max_depth": [3, 5],
    }

    tsplit = TimeSeriesSplit(n_splits=5)

    # ---- Randomized search cv
    random_search = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(),
        param_distributions=param_grid,
        scoring="accuracy",
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
    """
    info: define model pipe for support vector machine
    input: n x n array input features, response label
    output: fitted predictor pipe for support vector machine
    """
    # ---- Pipeline for Random Forest tuning
    svm_pipe = Pipeline(
        [
            ("preprocess", pipeline_ops()),
            ("scaler", StandardScaler()),
            ("regressor", SVC(C=0.1, kernel="rbf", gamma="scale")),
        ]
    )

    # ---- Model fitting
    svm = svm_pipe.fit(X_train, y_train.values.ravel())
    return svm


def main():
    """
    info: entry point into script
    input: none
    output: none
    """
    # --- start data prep workflow
    X_train, y_train, X_test, y_test = data_prep()

    # --- start logistic model workflow
    estimator_log = model_pipe_log(
        X_train=X_train,
        y_train=y_train,
    )

    logging.info("Building logistic model --- Done")

    # ---- start evaluation workflows
    log_train = eval_metrics(model=estimator_log, X=X_train, y_true=y_train)
    log_test = eval_metrics(model=estimator_log, X=X_test, y_true=y_test)

    # --- log results
    logging.info(f"\n log_train: {log_train}")
    logging.info(f"n log_train: {log_test}")

    # --- serialize model
    with open("./models_all_params/log.pickle", "wb") as f:
        pickle.dump(estimator_log, f)

    #  --- start random forest workflow
    estimator_rf = model_pipe_rf(
        X_train=X_train,
        y_train=y_train,
    )
    logging.info("Building random forest model --- Done")

    # --- start evaluation workflow
    rf_train = eval_metrics(model=estimator_rf, X=X_train, y_true=y_train)
    rf_test = eval_metrics(model=estimator_rf, X=X_test, y_true=y_test)

    # --- log metric results
    logging.info(f"\n rf_train: {rf_train}")
    logging.info(f"n rf_test: {rf_test}")

    # --- serialize random forest model
    with open("./models_all_params/rf.pickle", "wb") as f:
        pickle.dump(estimator_rf, f)

    #  --- start decision tree workflow
    estimator_dtc = model_pipe_dtc(
        X_train=X_train,
        y_train=y_train,
    )
    logging.info("Building decision tree model --- Done")

    # --- start evaluation workflow
    dtc_train = eval_metrics(model=estimator_dtc, X=X_train, y_true=y_train)
    dtc_test = eval_metrics(model=estimator_dtc, X=X_test, y_true=y_test)

    # --- log metric results
    logging.info(f"\n dtc_train: {dtc_train}")
    logging.info(f"n dtc_test: {dtc_test}")

    # --- serialize model
    with open("./models_all_params/dtc.pickle", "wb") as f:
        pickle.dump(estimator_dtc, f)

    # --- start svm workflow
    estimator_svm = model_pipe_svm(
        X_train=X_train,
        y_train=y_train,
    )
    logging.info("Building SVM model --- Done")

    # --- start evaluation workflows
    svm_train = eval_metrics(model=estimator_svm, X=X_test, y_true=y_test)
    svm_test = eval_metrics(model=estimator_svm, X=X_test, y_true=y_test)

    # --- log results
    logging.info(f"\n svm_train: {svm_train}")
    logging.info(f"n svm_test: {svm_test}")

    #  --- serialize model
    with open("./models_all_params/svm.pickle", "wb") as f:
        pickle.dump(estimator_svm, f)

    #collect all results
    metric_result = pd.DataFrame(
        [log_train, log_test, rf_train, rf_test, svm_train, svm_test],
        index=["log_train", "log_test", "rf_train", "rf_test", "svm_train", "svm_test"],
    )
    #write metrics out
    metric_result.to_csv("./results/metric_result.csv")

    return logging.info("\n all jobs completed")


if __name__ == "__main__":
    main()
