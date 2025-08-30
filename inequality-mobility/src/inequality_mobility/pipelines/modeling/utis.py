import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedGroupKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def modeling(estimator, X_train, X_test, y_train, y_test, params, map_class):
    # Fit and Training model
    model = estimator(**params)
    model.fit(X_train, y_train)
    # Predict test and train datasets
    y_pred_test = model.predict(X_test)
    # Plot Confusion Matrix
    _, ax = plt.subplots(1, figsize=(12, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    labels = map_class.keys()
    # Classification Report
    print(classification_report(y_test, y_pred_test, target_names=labels))
    return model


def get_validation(X, y):
    return train_test_split(X, y, test_size=0.25, random_state=42)


def get_groups(X_train):
    return X_train.reset_index()['identifica_pessoa']


def kfold_result(X, y, n_splits, Model, model_args, fit_args = {}, nn=False):
    accs = []
    groups = get_groups(X)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for k, (train_idx, test_idx) in enumerate(sgkf.split(X.values, y.values, groups=groups)):
        print(f"KFold: {k+1}")
        cv_X_train, cv_y_train = X.values[train_idx], y.values[train_idx]
        cv_X_test, cv_y_test = X.values[test_idx], y.values[test_idx]
        clf = Model(**model_args)
        clf.fit(cv_X_train, cv_y_train, **fit_args)
        if nn:
            y_pred = np.argmax(clf.predict(cv_X_test), axis=1)
        else:
            y_pred = clf.predict(cv_X_test)
        acc = accuracy_score(cv_y_test, y_pred)
        accs.append(acc)
    print(accs)
    return np.mean(accs)

