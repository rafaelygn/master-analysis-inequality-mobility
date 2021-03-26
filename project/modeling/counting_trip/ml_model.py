import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from utils import valid_model

PATH_PROJECT = "/home/yoshraf/projects/mestrado/"

X_train = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/X_train.parquet")
X_test = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/X_test.parquet")
y_train = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/y_train.parquet")
y_test = pd.read_parquet(f"{PATH_PROJECT}data/counting_trips/y_test.parquet")


def rf_modeling(X_train, X_test, y_train, y_test, params):
    reg = RandomForestRegressor(**params)
    reg.fit(X_train, y_train.values.ravel())
    y_pred_train = reg.predict(X_train)
    y_pred_test = reg.predict(X_test)
    feats = {}
    for feature, importance in zip(X_train.columns, reg.feature_importances_):
        feats[feature] = importance * 100
    df_imp = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Importance'}).sort_values("Importance", ascending=False)
    print("Main Features:")
    print(df_imp.head(10))
    valid_model(y_train, y_test, y_pred_train, y_pred_test)


rf_params = {"n_estimators": 800, "max_depth": 4, "random_state": 0, "min_samples_leaf": 30, "max_features": "sqrt", "max_samples": .8}
rf_modeling(X_train, X_test, y_train, y_test, rf_params)
rf_modeling(X_train, X_test, y_train, y_test, {})
