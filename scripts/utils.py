import os
from pathlib import Path
import seaborn as sns
import mlflow
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

mlflow.set_tracking_uri('http://0.0.0.0:5000')

def get_project_paths():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    return {
        "project_root": PROJECT_ROOT,
        "raw_dir": PROJECT_ROOT / paths["raw_data"],
        "processed_dir": PROJECT_ROOT / paths["processed_data"],
        "models_dir": PROJECT_ROOT / paths["models_dir"],
        "scripts_dir": PROJECT_ROOT / paths["scripts"],
        "image_dir": PROJECT_ROOT / paths["image_dir"],
        "vectors_dir": PROJECT_ROOT / paths["vectors_dir"],
        "logs_dir": PROJECT_ROOT / paths["logs_dir"],
        "optuna_results": PROJECT_ROOT / paths["optuna_results"],
    }

def get_project_root():
    """Возвращает абсолютный путь к корню проекта."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))




def log_confusion_matrix(all_preds, all_labels, model_name_tag):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    # Вычисление матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)

    # Визуализация матрицы ошибок
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plot_path = paths["image_dir"]/f"{model_name_tag}_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)  # Логирование матрицы ошибок
    plt.close()

def plot_losses(train_losses, val_losses, model_name_tag)->str:
    """
    Функция готовит графики для сохранения/передачи в mlflow
    :param train_losses: Функция потерь при обучении
    :param val_losses: Функция потерь при валидации
    :param model_name_tag:  название текущего графика
    :return: Путь к изображению
    """
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


    plot_path =  paths["image_dir"] /f"training_loss_curve_{model_name_tag}.png"
    plt.savefig(plot_path)
    plt.close()

    return plot_path  # Возвращаем путь к сохранённому файлу