import keras
import numpy as np
import optuna
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adadelta, Adam
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from optuna.samplers import TPESampler


from .utis import get_validation, kfold_result

SAVE_RESULTS = '/home/yoshraf/projects/master-analysis-inequality-mobility/inequality-mobility/'
NCALLS = 50
NFOLDS = 4
SAMPLER = TPESampler(seed=42)


def modeling(estimator, X_train, X_test, y_train, y_test, params, map_class, save_report=False):
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
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return model


def node_modeling(X_train, X_test, y_train, y_test, map_class):
    # Mapping y_class to a numerical
    y_train = y_train["Modo Combinado"].map(map_class)
    y_test = y_test["Modo Combinado"].map(map_class)
    # Training
    clf_logit = modeling(LogisticRegression, X_train, X_test,
                         y_train, y_test, {}, map_class, 'logit_dummy')
    clf_rf = modeling(RandomForestClassifier, X_train, X_test,
                      y_train, y_test, {}, map_class, 'rf_dummy')
    return clf_rf, clf_logit

#################################################################################################
#################################################################################################
# 
#    LOGIT
# 
#################################################################################################
#################################################################################################


def node_logit(X_train, X_test, y_train, y_test, map_class, scaler=False):
    # Mapping y_class to a numerical
    y_train = y_train["modo_principal_da_viagem"].map(map_class)
    y_test = y_test["modo_principal_da_viagem"].map(map_class)
    if scaler:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    logit_params = {
        'penalty': None,
        # 'tol': 1e-4,
        'random_state': 42,
        'max_iter': 500,
        'multi_class': 'multinomial',
    }
    clf_logit = modeling(LogisticRegression, X_train, X_test,
                         y_train, y_test, logit_params, map_class, 'logit_dummy')
    return clf_logit


#################################################################################################
#################################################################################################
# 
#    DECISION TREE
# 
#################################################################################################
#################################################################################################

def modeling_scikit(X, y, Model, model_args):
    cv_metric = kfold_result(X, y, NFOLDS, Model, model_args)
    return cv_metric


def objective_tree(trial, X, y):
    model_args = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "max_depth": trial.suggest_int("max_depth", 1, 30),    
        "min_samples_split": trial.suggest_int("min_samples_split", 1, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "random_state": 42,
    }
    return modeling_scikit(X, y, DecisionTreeClassifier, model_args)


def node_tree(X_train, X_test, y_train, y_test, map_class, save_report='tree'):
    
    study = optuna.create_study(sampler=SAMPLER, direction='maximize')
    study.optimize(
        lambda trial: objective_tree(trial, X_train, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)
    clf = DecisionTreeClassifier(**study.best_params)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)


#################################################################################################
#################################################################################################
# 
#    RANDOM FOREST
# 
#################################################################################################
#################################################################################################

def objective_rf(trial, X, y):
    model_args = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "max_depth": trial.suggest_int('max_depth', 5, 50),
        "criterion": trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 11),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 11),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "n_jobs":1,
        "random_state": 42
    }
    return modeling_scikit(X, y, RandomForestClassifier, model_args)


def node_rf(X_train, X_test, y_train, y_test, map_class, save_report='rf'):
    study = optuna.create_study(sampler=SAMPLER, direction='maximize')
    study.optimize(
        lambda trial: objective_rf(trial, X_train, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)
    clf = RandomForestClassifier(**study.best_params)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)


#################################################################################################
#################################################################################################
# 
#    SVM
# 
#################################################################################################
#################################################################################################

def objective_svm(trial, X, y):
    n_samples = 10_000
    model_args = {
        "C": trial.suggest_float("C", 1e-2, 3),
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "sigmoid", 'poly']),
        "random_state": 42,
    }
    scaler = RobustScaler()
    X_r = scaler.fit_transform(X)
    X = pd.DataFrame(X_r, index=X.index)
    dataset = X.assign(y=y).sample(n_samples, random_state=42)
    return modeling_scikit(dataset[X.columns], dataset[['y']], SVC, model_args)


def node_svm(X_train, X_test, y_train, y_test, map_class, save_report='svm'):
    study = optuna.create_study(sampler=SAMPLER, direction='maximize')
    study.optimize(
        lambda trial: objective_svm(trial, X_train, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = SVC(**study.best_params)
    clf.fit(X_train, y_train)
    
    y_pred_test = clf.predict(X_test)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)

#################################################################################################
#################################################################################################
# 
#    LIGHTGBM
# 
#################################################################################################
#################################################################################################


def modeling_early_stop(X, y, Model, model_args):
    X_train, X_validation, y_train, y_validation = get_validation(X, y)
    fit_args = {
        "eval_set": [(X_validation.values, y_validation.values)],
        "early_stopping_rounds":2,
        "verbose":False
    }
    cv_metric = kfold_result(X_train, y_train, NFOLDS, Model, model_args, fit_args)
    return cv_metric


def objective_lgbm(trial, X, y):
    model_args = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 100),
        "num_leaves": trial.suggest_int("num_leaves", 3, 90),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1, log=True),
        "colsample_bytree": trial.suggest_float('colsample_bytree', .5, 1),
        "subsample": trial.suggest_float('subsample', .5, 1),
        "max_depth": trial.suggest_int("max_depth", -1, 15),
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 1
    }
    return modeling_early_stop(X, y, LGBMClassifier, model_args)


def node_lgbm(X_train, X_test, y_train, y_test, map_class, save_report='lgbm'):
    study = optuna.create_study(sampler=SAMPLER, direction='maximize')
    study.optimize(
        lambda trial: objective_lgbm(trial, X_train, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)
    clf = LGBMClassifier(**study.best_params)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)

#################################################################################################
#################################################################################################
# 
#    XGBOOST
# 
#################################################################################################
#################################################################################################


def objective_xgb(trial, X, y):
    model_args = {
        "n_estimators": 250,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "colsample_bytree": trial.suggest_float('colsample_bytree', .5, 1),
        "subsample": trial.suggest_float('subsample', .5, 1),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "random_state": 42,
        "nthread": -1
    }
    return modeling_early_stop(X, y, XGBClassifier, model_args)


def node_xgb(X_train, X_test, y_train, y_test, map_class, save_report='xgb'):
    study = optuna.create_study(sampler=SAMPLER, direction='maximize')
    study.optimize(
        lambda trial: objective_xgb(trial, X_train, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)
    clf = LGBMClassifier(**study.best_params)
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)


#################################################################################################
#################################################################################################
# 
#    TENSORFLOW
# 
#################################################################################################
#################################################################################################



def tf_model_arch(layer_sizes, n_input, n_output, dropout_rate, activation, learning_rate, Algo):
    inputs = keras.Input(shape=n_input, name='Input')
    x = inputs
    for i, layer_size in enumerate(layer_sizes):
        x = Dense(layer_size, activation=activation, name=f"Layer_{i+1}")(x)
        x = Dropout(dropout_rate, seed=42)(x)
    outputs = Dense(n_output, activation='softmax', name='Output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=Algo(learning_rate),
        metrics=["accuracy"]
    )
    return model


def objective_ann(trial, X, y):
    # Define the search space for the number of neurons in each layer
    n_layers = trial.suggest_int('n_layers', 1, 5)
    layer_sizes = []
    for i in range(n_layers):
        layer_size = trial.suggest_int(f'layer_size_{i+1}', 16, 256)
        layer_sizes.append(layer_size)
    
    # NN args
    args_arc_nn = {
        "layer_sizes": layer_sizes,
        "n_input": X.shape[1],
        "n_output": trial.suggest_int("n_output", 10, 35),
        "dropout_rate": trial.suggest_float('dropout_rate', 0, .7),
        "activation": trial.suggest_categorical('activation', ['relu', 'tanh', 'elu', 'sigmoid']),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        "Algo": trial.suggest_categorical("Algo", [Adam, Adadelta, SGD])
    }
    # Fit args
    early_stopping = EarlyStopping(patience=trial.suggest_int("patience", 1, 5), min_delta=1e-4)
    fit_args = {
        'epochs': 100,
        'batch_size': trial.suggest_int("batch_size", 16, 256),
        'validation_split': 0.2,
        'callbacks': [early_stopping],
        'verbose': -1
    }
    return kfold_result(X, y, NFOLDS, tf_model_arch, args_arc_nn, fit_args, True)


def node_ann(X_train, X_test, y_train, y_test, map_class, save_report='ann'):
    # Scaling the Data
    scaler = RobustScaler()
    X_train_r = scaler.fit_transform(X_train)
    X_test_r = scaler.transform(X_test)
    X_train_r = pd.DataFrame(X_train_r, index=X_train.index)
    # Label Encoder
    l_encoder = LabelEncoder()
    y_train = pd.DataFrame(l_encoder.fit_transform(y_train), index=y_train.index)
    y_test = pd.DataFrame(l_encoder.transform(y_test), index=y_test.index)
    
    study = optuna.create_study(sampler=SAMPLER,direction='maximize')
    study.optimize(
        lambda trial: objective_ann(trial, X_train_r, y_train), 
        n_trials=NCALLS,
        n_jobs=1,
        show_progress_bar=True
    )
    print("optimum value: ", study.best_value)
    print("Optimal parameters: ", study.best_params)

    # Define the search space for the number of neurons in each layer
    opt_param = study.best_params.copy()
    n_layers = opt_param.pop('n_layers')
    layer_sizes = []
    for i in range(n_layers):
        layer_size = opt_param.pop(f'layer_size_{i+1}')
        layer_sizes.append(layer_size)
    opt_param['layer_sizes'] = layer_sizes
    opt_param['n_input'] = X_train_r.shape[1]
    fit_args = {
        'epochs': 100,
        'validation_split': .2,
        'verbose': -1,
    }
    early_stopping = EarlyStopping(patience=opt_param.pop('patience'), min_delta=1e-4)
    fit_args['batch_size'] = opt_param.pop('batch_size')
    fit_args['callbacks'] = early_stopping

    clf = tf_model_arch(**opt_param)
    clf.fit(X_train_r, y_train, **fit_args)
    y_pred_test = np.argmax(clf.predict(X_test_r), axis=1)
    labels = map_class.keys()
    print(classification_report(y_test, y_pred_test, target_names=labels))
    if save_report:
        report = classification_report(
            y_test, y_pred_test, target_names=labels, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(
            SAVE_RESULTS + f'docs/ml_results/{save_report}.csv')
    return (clf, study, report)