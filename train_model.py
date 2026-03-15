import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

TARGET_COL = "Survived"


def scale_frame(frame):
    df = frame.copy()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X.values)

    return X_scale, y.values, scaler


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    return accuracy, precision, recall, f1


def train():
    df = pd.read_csv("./df_clear.csv")
    X, y, scaler = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    mlflow.set_experiment("titanic_classifier")

    with mlflow.start_run():
        model = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        accuracy, precision, recall, f1 = eval_metrics(y_val, y_pred)

        mlflow.log_param("model_type", "SGDClassifier")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("tol", 1e-3)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        joblib.dump(model, "sgd_titanic.pkl")
        joblib.dump(scaler, "scaler_titanic.pkl")

        print("Training finished successfully")
        print(f"accuracy: {accuracy}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {f1}")
