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
    dag_id="optuna_search",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["hugging"]
) as dag:

    optuna_search = BashOperator(
        task_id="optuna_search",
        bash_command="cd /home/saatarko/PycharmProjects/CNN_learn && dvc repro optuna_search_stage",
        doc_md = "**Проведение исследований при классификации**"
    )


    optuna_search