import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import mlflow
import optuna
import pandas as pd
import torch
from adabelief_pytorch import AdaBelief
from optuna.samplers import GridSampler
from ranger_adabelief import RangerAdaBelief
from PIL import Image
import io
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, Dataset
import torch.nn.functional as F
from torchvision import transforms

from utils import get_project_paths, plot_losses
from task_registry import main, task

paths = get_project_paths()
# Настройка отдельного логгера для неудачных трaйлов
log_path = paths["logs_dir"] / "optuna_failures.log"

error_logger = logging.getLogger("optuna_failures")
error_logger.setLevel(logging.WARNING)
handler = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
error_logger.addHandler(handler)

mlflow.set_tracking_uri('http://localhost:5000')

class ByteImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Достаём байты
        byte_data = row['image']['bytes']
        image = Image.open(io.BytesIO(byte_data)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row['labels']
        return image, label


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))  # softplus(x) = ln(1 + e^x)


def get_optimizer(name, model_parameters, lr=1e-3):
    name = name.lower()

    if name == "adam":
        return optim.Adam(model_parameters, lr=lr)
    elif name == "adamw":
        return optim.AdamW(model_parameters, lr=lr)
    elif name == "rmsprop":
        return optim.RMSprop(model_parameters, lr=lr)
    elif name == "ranger":
        if RangerAdaBelief is None:
            raise ImportError(
                "ranger-adabelief package is not installed. Install it with `pip install ranger-adabelief`")
        return RangerAdaBelief(model_parameters, lr=lr)
    elif name == "adabelief":
        if AdaBelief is None:
            raise ImportError(
                "adabelief-pytorch package is not installed. Install it with `pip install adabelief-pytorch`")
        return AdaBelief(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: {name}")



class MyCNN(nn.Module):
    def __init__(self, activation='relu', dropout=None):
        super().__init__()

        activation = activation.lower()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "swish":
            self.activation = Swish()
        elif activation == "mish":
            self.activation = Mish()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.use_dropout = dropout is not None and dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Размер после пулингов для входа 224x224:
        # 224 -> 112 -> 56 -> 28
        self.fc = nn.Linear(28*28*64, 1000)
        self.out = nn.Linear(1000, 1)  # 1 выход — логит для бинарной классификации

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))
        if self.use_dropout and self.training:
            x = self.dropout(x)
        x = self.pool3(self.activation(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.use_dropout and self.training:
            x = self.dropout(x)

        features = x
        logits = self.out(features)  # выход без сигмоиды

        return logits, features


def get_data_loader(split: str, augment: bool = False, batch_size: int = 256):
    """
    Возвращает DataLoader для заданного типа данных (train/val/test).

    Параметры:
    - split: 'train', 'val' или 'test'
    - augment: применять ли аугментации (только для train)
    - batch_size: размер батча

    Возвращает:
    - torch.utils.data.DataLoader
    """
    paths = get_project_paths()
    df = pd.read_parquet(paths["raw_dir"] / f"{split}.parquet")

    IMG_SIZE = 224
    RESIZE_TO = 256  # ресайз чуть больше для кропа

    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225]
    )

    if augment and split == "train":
        transform = transforms.Compose([
            transforms.Resize(RESIZE_TO),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalization,
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(RESIZE_TO),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            normalization
        ])

    dataset = ByteImageDataset(df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loader

def cnn_train(model, criterion, optimizer_name, model_name_tag, num_epochs=30, device=None, weight_decay=0.0, lr=1e-3):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    train_loader = get_data_loader("train", augment=True)
    val_loader = get_data_loader("val")

    # Создаем оптимизатор с регуляризацией weight_decay
    optimizer = get_optimizer(optimizer_name, model.parameters(), lr=lr)
    # Обновим weight_decay, если нужно (если оптимизатор его поддерживает)
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = weight_decay

    best_val_loss = float("inf")
    best_model_state = None

    epoch_train_losses = []
    epoch_val_losses = []

    model = model.to(device)

    model_path = paths["models_dir"] / f"{model_name_tag}_final_model.pt"
    metrics_path = paths["models_dir"] / f"{model_name_tag}_train_metrics.json"

    with mlflow.start_run(nested=True):
        mlflow.log_param("model_name_tag", model_name_tag)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", 256)
        mlflow.log_param("seed", 42)
        mlflow.log_param("weight_decay", weight_decay)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output, _ = model(X_batch)
                loss = criterion(output.squeeze(), y_batch.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    output, _ = model(X_batch)
                    loss = criterion(output.squeeze(), y_batch.float())
                    val_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)

            epoch_train_losses.append(avg_train)
            epoch_val_losses.append(avg_val)

            mlflow.log_metric("train_loss", avg_train, step=epoch)
            mlflow.log_metric("val_loss", avg_val, step=epoch)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = model.state_dict()

            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        torch.save(best_model_state, model_path)
        mlflow.pytorch.log_model(model, model_name_tag)

        plot_path = plot_losses(epoch_train_losses, epoch_val_losses, model_name_tag)
        mlflow.log_artifact(str(plot_path))

        train_metrics = {
            "best_val_loss": best_val_loss,
            "final_train_loss": epoch_train_losses[-1],
            "final_val_loss": epoch_val_losses[-1],
            "num_epochs": num_epochs,
            "weight_decay": weight_decay
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(train_metrics, f, indent=4)

    return best_model_state



def objective(trial):
    try:
        activation = trial.suggest_categorical("activation",
                                               ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'AdamW', 'RMSProp', 'Ranger', 'AdaBelief'])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        num_epochs = 30

        criterion = nn.BCEWithLogitsLoss()
        model = MyCNN(activation=activation, dropout=dropout)

        model_name_tag = f"{activation}"
        if dropout > 0: model_name_tag += "_drop"
        model_name_tag += f"_{optimizer_name}_wd{weight_decay:.5f}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

        best_model_state = cnn_train(
            model=model,
            criterion=criterion,
            optimizer_name=optimizer_name,
            model_name_tag=model_name_tag,
            num_epochs=num_epochs,
            device=device,
            weight_decay=weight_decay,
            lr=learning_rate
        )

        model.load_state_dict(best_model_state)
        model.eval()

        # Валидация + логирование
        val_loader = get_data_loader("val")
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output, _ = model(X_batch)
                loss = criterion(output.squeeze(), y_batch.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)

        trial_id = trial.number
        params = trial.params

        result = {
            "trial": trial_id,
            "params": params,
            "val_loss": val_loss
        }

        # Пусть путь к папке для результатов будет фиксирован, например:
        results_dir = Path("optuna_results")
        results_dir.mkdir(exist_ok=True)

        # Записываем в отдельный файл:
        with open(results_dir / f"trial_{trial_id}.json", "w") as f:
            json.dump(result, f, indent=4)

        with mlflow.start_run(nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metric(f"val_loss", val_loss)
            mlflow.pytorch.log_model(model, "model")

        return val_loss

    except Exception as e:
        error_logger.warning(f"Trial failed with params: {trial.params} -> {e}")
        return float("inf")






@task("data:run_optuna_search")
def run_optuna_search():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()


    search_space = {
        "activation": ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"],
        "optimizer": ['Adam', 'AdamW', 'RMSProp', 'Ranger', 'AdaBelief'],
        "dropout": [0.0, 0.2],
        "loss_fn": ["BCEWithLogitsLoss"],
        "weight_decay": [1e-4, 1e-3],
        "learning_rate": [1e-6, 1e-2],
    }

    sampler = GridSampler(search_space)
    study = optuna.create_study(sampler=sampler, direction="minimize")

    with mlflow.start_run(run_name=f"Class_GridSearch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("search_type", "GridSampler")

        study.optimize(lambda trial: objective(trial,), n_trials=len(sampler._all_grids))

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_loss", study.best_value)

        result_path = paths["models_dir"] / "class_optuna_best_result.json"
        mlflow.log_artifact(result_path)

        optuna_path = paths["models_dir"] / "class_optuna_study.pkl"
        mlflow.log_artifact(optuna_path)

    return study










































































if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()


    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
