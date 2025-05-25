from datetime import datetime
from airflow.models import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === DAG конфигурация ===
with DAG(
    dag_id="train_best_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["hugging"]
) as dag:

    train_best_model = BashOperator(
        task_id="train_best_model",
        bash_command="cd /home/saatarko/PycharmProjects/CNN_learn && dvc repro train_best_model_stage",
        doc_md = "**Обучение лучшей полученной модели*"
    )
