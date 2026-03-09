from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
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
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, y.values, scaler


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv("/home/mint/airflow/dags/df_clear.csv")

    X, y, scaler = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    model = SGDRegressor(
        random_state=42,
        alpha=0.01,
        l1_ratio=0.1,
        penalty="elasticnet",
        loss="squared_error",
        fit_intercept=True,
        max_iter=1000,
        tol=1e-3
    )

    mlflow.set_experiment("zomato_final_order_value")

    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        rmse, mae, r2 = eval_metrics(y_val, y_pred)

        mlflow.log_param("alpha", model.alpha)
        mlflow.log_param("l1_ratio", model.l1_ratio)
        mlflow.log_param("penalty", model.penalty)
        mlflow.log_param("loss", model.loss)
        mlflow.log_param("fit_intercept", model.fit_intercept)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        joblib.dump(model, "/home/mint/airflow/dags/zomato_model.pkl")
        joblib.dump(scaler, "/home/mint/airflow/dags/zomato_scaler.pkl")

        print("Training finished successfully")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")
