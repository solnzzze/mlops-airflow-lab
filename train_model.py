import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train

DATA_FILE = "Titanic-Dataset.csv"
CLEAR_FILE = "df_clear.csv"
TARGET_COL = "Survived"


def download_data():
    df = pd.read_csv(DATA_FILE)
    print("df:", df.shape)
    return True


def clear_data():
    df = pd.read_csv(DATA_FILE)

    cat_columns = ["Sex", "Embarked"]
    num_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]

    df = df.drop(columns=drop_columns, errors="ignore")

    df = df.dropna(subset=[TARGET_COL])

    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    for col in num_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df = df.reset_index(drop=True)
    df.to_csv(CLEAR_FILE, index=False)
    return True


dag_titanic = DAG(
    dag_id="train_titanic_pipe",
    start_date=datetime(2025, 2, 3),
    schedule=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data,
    task_id="download_titanic",
    dag=dag_titanic
)

clear_task = PythonOperator(
    python_callable=clear_data,
    task_id="clear_titanic",
    dag=dag_titanic
)

train_task = PythonOperator(
    python_callable=train,
    task_id="train_titanic",
    dag=dag_titanic
)

download_task >> clear_task >> train_task
