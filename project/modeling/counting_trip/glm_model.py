import pandas as pd
import statsmodels.api as sm

from utils import valid_model

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"

X_train = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/X_train.parquet")
X_test = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/X_test.parquet")
y_train = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/y_train.parquet")
y_test = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/y_test.parquet")


def lr_model(X_train, X_test, y_train, y_test):
    # Add Contante Feature
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    # Define and fit Model
    res = sm.OLS(y_train.values.ravel(), X_train).fit()
    # Print report
    print(res.summary())
    # Predict train and test
    y_pred_train = res.get_prediction(X_train).summary_frame()["mean"].values
    y_pred_test = res.get_prediction(X_test).summary_frame()["mean"].values
    # Report regression metrics
    valid_model(y_train, y_test, y_pred_train, y_pred_test)


def gl_model(X_train, X_test, y_train, y_test, family):
    # Add Contante Feature
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    # Define and fit Model
    res = sm.GLM(y_train, X_train, family=family).fit()
    # Print report
    print(res.summary())
    # Predict train and test
    y_pred_train = res.get_prediction(X_train).summary_frame()["mean"].values
    y_pred_test = res.get_prediction(X_test).summary_frame()["mean"].values
    # Report regression metrics
    valid_model(y_train, y_test, y_pred_train, y_pred_test)


lr_model(X_train, X_test, y_train, y_test)
gl_model(X_train, X_test, y_train, y_test, sm.families.Poisson())
gl_model(X_train, X_test, y_train, y_test, sm.families.Gamma())
gl_model(X_train, X_test, y_train, y_test, sm.families.NegativeBinomial())
