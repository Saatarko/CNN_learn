.PHONY: format lint clean_cashe init start stop logs clean create-mlflow-db ps init-env

# Форматирование: isort → autoflake → black
format:
	isort .
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r .
	black .

# Только сортировка импортов
isort:
	isort .

# Только black
black:
	black .

# Только удаление неиспользуемых импортов
autoflake:
	autoflake --in-place --remove-all-unused-imports --remove-unused-variables -r .

# Проверка соответствия PEP8 и другим правилам
lint:
	flake8 .

# Очистка временных и кэш-файлов
clean_cashe:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete


init:
	@echo "Запуск инициализации Airflow..."
	./init_airflow.sh


start:
	docker compose up -d
	sudo chown -R $(AIRFLOW_UID):$(AIRFLOW_UID) ./airflow_home/dags
	sudo chmod -R ug+rwX ./airflow_home/dags


stop:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker ps -a

clean:
	docker compose down -v
	sudo rm -rf airflow_home/logs/* airflow_home/plugins/* airflow_home/dags/*
	sudo chown -R $(shell id -u):$(shell id -g) airflow_home/logs airflow_home/plugins airflow_home/dags

create-mlflow-db:
	docker exec -i cnn_learn-postgres-1 psql -U airflow -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = 'mlflow_db'" | grep -q 1 || \
	docker exec -i cnn_learn-postgres-1 psql -U airflow -d postgres -c "CREATE DATABASE mlflow_db;"

init-env:
	echo "POSTGRES_USER=airflow" > .env
	echo "POSTGRES_PASSWORD=airflow" >> .env
	echo "AIRFLOW_DB=airflow_db" >> .env
	echo "MLFLOW_DB=mlflow_db" >> .env
	echo "MLFLOW_BACKEND_URI=postgresql://airflow:airflow@postgres:5432/mlflow_db" >> .env
	echo "ARTIFACT_ROOT=/mlflow/artifacts" >> .env
	echo "AIRFLOW_USERNAME=admin" >> .env
	echo "AIRFLOW_PASSWORD=admin" >> .env
	echo "AIRFLOW_FIRSTNAME=Admin" >> .env
	echo "AIRFLOW_LASTNAME=User" >> .env
	echo "AIRFLOW_EMAIL=saatarko@tut.by" >> .env
	echo "AIRFLOW_UID=$$(id -u)" >> .env
