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
    dag_id="evaluate_best_model",
    start_date=datetime(2024, 1, 1),
    schedule=None,  # или cron выражение
    catchup=False,
    tags=["hugging"]
) as dag:

    evaluate_best_model = BashOperator(
        task_id="evaluate_best_model",
        bash_command="cd /home/saatarko/PycharmProjects/CNN_learn && dvc repro evaluate_best_model_stage",
        doc_md = "**Предикты**"
    )


    evaluate_best_model