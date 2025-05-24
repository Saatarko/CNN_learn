#!/bin/bash
set -e

# Загружаем переменные окружения
set -a
source .env
set +a

# Проверка переменных окружения
echo "Создание пользователя‑админа..."
if [[ -z "$AIRFLOW_USERNAME" || -z "$AIRFLOW_PASSWORD" || -z "$AIRFLOW_FIRSTNAME" || -z "$AIRFLOW_LASTNAME" || -z "$AIRFLOW_EMAIL" ]]; then
  echo "❌ Ошибка: Не заданы переменные окружения для пользователя Airflow (AIRFLOW_USERNAME, AIRFLOW_PASSWORD, AIRFLOW_FIRSTNAME, AIRFLOW_LASTNAME, AIRFLOW_EMAIL)"
  exit 1
fi

airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin