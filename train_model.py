from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDRegressor
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib

TARGET_COL = "final_order_value"


def scale_frame(frame):
    df = frame.copy()

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    scaler = StandardScaler()
    power_trans = PowerTransformer()

    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, power_trans, scaler


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("df_clear.csv")

    X, Y, power_trans, scaler = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=42
    )

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.01, 0.05, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    mlflow.set_experiment("zomato_final_order_value")

    with mlflow.start_run():
        base_model = SGDRegressor(random_state=42)
        clf = GridSearchCV(base_model, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.reshape(-1))

        best = clf.best_estimator_

        y_pred = best.predict(X_val)

        y_pred_real = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_val_real = power_trans.inverse_transform(y_val)

        rmse, mae, r2 = eval_metrics(y_val_real, y_pred_real)

        mlflow.log_param("alpha", best.alpha)
        mlflow.log_param("l1_ratio", best.l1_ratio)
        mlflow.log_param("penalty", best.penalty)
        mlflow.log_param("loss", best.loss)
        mlflow.log_param("fit_intercept", best.fit_intercept)
        mlflow.log_param("epsilon", best.epsilon)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "model", signature=signature)

        joblib.dump(best, "zomato_model.pkl")
        joblib.dump(scaler, "zomato_scaler.pkl")
        joblib.dump(power_trans, "zomato_power_transformer.pkl")

        print("Training finished successfully")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
