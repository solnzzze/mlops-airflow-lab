import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train


DATA_FILE = "csao_session_dataset.csv"
CLEAR_FILE = "df_clear.csv"
TARGET_COL = "final_order_value"


def download_data():
    df = pd.read_csv(DATA_FILE)
    print("Dataset loaded:", df.shape)
    print("Columns:", df.columns.tolist())
    return True


def clear_data():
    df = pd.read_csv(DATA_FILE)

    drop_columns = [
        "session_id",
        "session_timestamp",
        "user_id",
        "restaurant_id",
        "restaurant_name",
        "base_cart_item_names",
        "recommended_addon_names",
        "actual_added_addon_names",
    ]

    cat_columns = [
        "user_segment",
        "user_city",
        "user_preferred_cuisine",
        "user_veg_preference",
        "user_preferred_addon_category",
        "restaurant_city",
        "restaurant_cuisine",
        "restaurant_type",
        "restaurant_online_order",
        "restaurant_price_tier",
        "restaurant_is_chain",
        "meal_time",
        "is_weekend",
        "has_offer",
        "weather_condition",
        "traffic_density",
        "is_festival_day",
        "delivery_zone",
        "base_cart_item_categories",
        "cart_has_drink",
        "cart_has_dessert",
        "cart_has_side",
        "recommended_addon_categories",
        "actual_added_addon_categories",
        "any_addon_added",
    ]

    num_columns = [
        "user_price_sensitivity",
        "user_order_frequency_30d",
        "user_avg_order_value",
        "user_recency_days",
        "num_past_orders_at_restaurant",
        "user_addon_acceptance_rate",
        "restaurant_rating",
        "restaurant_delivery_time_avg",
        "restaurant_avg_orders_per_day",
        "hour",
        "day_of_week",
        "estimated_delivery_time",
        "session_engagement_score",
        "base_cart_item_count",
        "base_cart_value",
        "cart_completion_score",
        "recommended_addon_prices",
        "actual_added_addon_count",
        "actual_added_addon_value",
        "avg_reco_score",
        "avg_reco_price_ratio",
        "avg_reco_popularity",
        "avg_reco_is_complementary",
    ]

    df = df.drop(columns=drop_columns, errors="ignore")

    df = df.dropna(subset=[TARGET_COL])

    for col in cat_columns:
        if col in df.columns:
            df[col] = df[col].fillna("missing").astype(str)

    for col in num_columns + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    encoder = OrdinalEncoder()
    existing_cat_columns = [col for col in cat_columns if col in df.columns]
    df[existing_cat_columns] = encoder.fit_transform(df[existing_cat_columns])

    df = df.reset_index(drop=True)
    df.to_csv(CLEAR_FILE, index=False)

    print("Clean dataset saved:", df.shape)
    return True


dag_zomato = DAG(
    dag_id="train_zomato_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval=timedelta(minutes=5),
    max_active_runs=1,
    catchup=False,
)

download_task = PythonOperator(
    python_callable=download_data,
    task_id="download_zomato",
    dag=dag_zomato
)

clear_task = PythonOperator(
    python_callable=clear_data,
    task_id="clear_zomato",
    dag=dag_zomato
)

train_task = PythonOperator(
    python_callable=train,
    task_id="train_zomato",
    dag=dag_zomato
)

download_task >> clear_task >> train_task
