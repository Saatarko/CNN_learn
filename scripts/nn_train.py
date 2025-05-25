import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import RandAugment
from tqdm import tqdm
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from adabelief_pytorch import AdaBelief
from optuna.samplers import GridSampler
from ranger_adabelief import RangerAdaBelief
from PIL import Image
import io
from torch import optim, nn
from torch.utils.data import DataLoader, random_split, Dataset, WeightedRandomSampler
import torch.nn.functional as F
from torchvision import transforms
import random
from utils import get_project_paths, plot_losses, log_confusion_matrix
from task_registry import main, task
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

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
    def __init__(self, activation='relu', dropout=None, spatial_drop=0.2):
        super().__init__()

        # --- активация ---
        act = activation.lower()
        if   act == "relu":      self.activation = nn.ReLU()
        elif act == "tanh":      self.activation = nn.Tanh()
        elif act == "sigmoid":   self.activation = nn.Sigmoid()
        elif act == "leaky_relu":self.activation = nn.LeakyReLU()
        elif act == "elu":       self.activation = nn.ELU()
        elif act == "swish":     self.activation = Swish()
        elif act == "mish":      self.activation = Mish()
        else: raise ValueError(f"Unknown activation: {activation}")

        # --- dropout'ы ---
        self.use_dropout = dropout is not None and dropout > 0
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)

        # Spatial-dropout-2D по каналам
        self.use_spatial = spatial_drop is not None and spatial_drop > 0
        if self.use_spatial:
            self.spatial_dropout = nn.Dropout2d(spatial_drop)

        # --- сверточные блоки ---
        self.conv1 = nn.Conv2d(3,  8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(28 * 28 * 64, 1000)
        self.out = nn.Linear(1000, 1)

        # для сохранения градиентов conv3_out
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pool1(self.activation(self.conv1(x)))
        x = self.pool2(self.activation(self.conv2(x)))

        if self.use_spatial and self.training:
            x = self.spatial_dropout(x)

        conv3_out = self.activation(self.conv3(x))
        conv3_out = self.pool3(conv3_out)

        # Регистрируем hook только если возможны градиенты
        if conv3_out.requires_grad:
            conv3_out.register_hook(self.activations_hook)

        x = torch.flatten(conv3_out, 1)
        x = self.fc(x)
        if self.use_dropout and self.training:
            x = self.dropout(x)

        features = x
        logits = self.out(features)
        return logits, features, conv3_out


def get_data_loader(split: str,
                    augment: bool = False,
                    batch_size: int = 256,
                    balanced: bool = False):
    paths = get_project_paths()
    df = pd.read_parquet(paths["raw_dir"] / f"{split}.parquet")

    IMG_SIZE, RESIZE_TO = 224, 256

    normalization = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std=[0.229,0.224,0.225])

    if augment and split == "train":
        transform = transforms.Compose([
            transforms.Resize(RESIZE_TO),
            RandAugment(),
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
            normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # фиксируем обе стороны
            transforms.ToTensor(),
            normalization
        ])

    dataset = ByteImageDataset(df, transform=transform)

    if balanced and split == "train":
        # считаем веса классов для balanced sampler
        labels = df["labels"].values
        class_sample_count = np.bincount(labels)
        weights = 1. / class_sample_count[labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"))

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


@task("data:evaluate_best_model")
def evaluate_best_model():

    paths = get_project_paths()
    model_path = paths["models_dir"] / "mish_best_final_model_new_vers.pt"
    activation = "mish"
    dropout = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyCNN(activation=activation, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader = get_data_loader("test")

    all_preds = []
    all_labels = []
    all_features = []

    fc_dir = paths["vectors_dir"] / "fc_features"
    fc_dir.mkdir(parents=True, exist_ok=True)

    conv3_dir = paths["vectors_dir"] / "conv3_out"
    conv3_dir.mkdir(parents=True, exist_ok=True)
    conv3_counter = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            inputs, labels = inputs.to(device), labels.to(device)

            logits, features, conv3_out = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            # all_features.append(features.cpu().numpy())  # (B, 1000)
            #
            # # Сохраняем fc1000 по одному
            # for j in range(features.shape[0]):
            #     idx = i * test_loader.batch_size + j
            #     vec = features[j].cpu().numpy()
            #     out_path = fc_dir / f"img{idx:05d}_fc1000.npy"
            #     np.save(out_path, vec)

            # Сохраняем первые 100 conv3_out поштучно
            if conv3_counter < 100:
                batch_limit = min(inputs.shape[0], 100 - conv3_counter)
                for j in range(batch_limit):
                    out_path = conv3_dir / f"img{conv3_counter:05d}_conv3.npy"
                    np.save(out_path, conv3_out[j].cpu().numpy())
                    conv3_counter += 1

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    acc = (all_preds == all_labels).mean()
    mlflow.log_metric("accuracy", acc)
    log_confusion_matrix(all_preds, all_labels, "mish_best_final_model_new_vers")

    np.save(paths["vectors_dir"] / "pred_mish_new_vers.npy", all_preds)
    np.save(paths["vectors_dir"] / "labels_mish_new_vers.npy", all_labels)

    # all_features = np.concatenate(all_features, axis=0)  # (N, 1000)
    # np.save(paths["vectors_dir"] / "test_fc1000_mish_new_vers.npy", all_features)
    # np.save(paths["vectors_dir"] / "test_labels_mish_new_vers.npy", all_labels)




@task("data:best_model_train")
def best_model_train():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml"):     # читалось только ради side-effect → убрал
        paths = get_project_paths()

    # --- гиперпараметры ------------------------------------------------------
    num_epochs = 60
    swa_start_epoch = 55  # было 40
    lr_base = 3e-4
    lr_max = 1e-3  # было 3e-3  ← мягче
    pct_start = 0.1  # раньше 0.3  ← короче рост
    weight_decay = 1e-4
    batch_size = 128
    mixup_prob = 0.5  # применяем только к половине батчей
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- модель ------------------------------------------------------------------
    model = MyCNN(
        activation='mish',
        dropout=None
    ).to(device)

    model.to(device)

    # --- данные --------------------------------------------------------------
    train_loader = get_data_loader("train", augment=True,
                                   batch_size=batch_size, balanced=True)
    val_loader   = get_data_loader("val", batch_size=batch_size)

    # --- class imbalance: pos_weight ----------------------------------------
    y_train = pd.read_parquet(paths["raw_dir"] / "train.parquet")["labels"].values
    class_cnt = np.bincount(y_train)  # [count_0, count_1]
    neg, pos = class_cnt[0], class_cnt[1]
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)

    # --- loss с label-smoothing (0.05) --------------------------------------
    def smooth_bce_loss(logits, targets, smoothing=0.05):
        targets = targets.float() * (1 - smoothing) + 0.5 * smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    # --- оптимайзер + One-Cycle scheduler -----------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=lr_base, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr_max,
        epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=pct_start
    )

    # --- SWA -----------------------------------------------------------------
    swa_model = torch.optim.swa_utils.AveragedModel(model, device=device)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=5e-5)  # чуть ниже

    # --- EMA (эксп. скользящее среднее весов) --------------------------------
    ema_decay = 0.999
    ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}

    # --- MLflow --------------------------------------------------------------
    run_params = dict(model="MyCNN+mish", epochs=num_epochs,
                      batch_size=batch_size, weight_decay=weight_decay)

    best_val_loss, best_state = float("inf"), None
    history = {"train":[], "val":[]}

    with mlflow.start_run(nested=True):
        mlflow.log_params(run_params)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)

                # ---------- MixUp (p = 0.5) -----------------------------------------
                if torch.rand(1).item() < mixup_prob:
                    lam = np.random.beta(0.4, 0.4)
                    idx = torch.randperm(xb.size(0))
                    mixed_x = lam * xb + (1 - lam) * xb[idx]
                    y_a, y_b = yb.float().unsqueeze(1), yb[idx].float().unsqueeze(1)
                    logits, _, _ = model(mixed_x)
                    loss = (lam * smooth_bce_loss(logits, y_a) +
                            (1 - lam) * smooth_bce_loss(logits, y_b))
                else:
                    logits, _, _ = model(xb)
                    loss = smooth_bce_loss(logits, yb.float().unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # --------- EMA update ----------------------------------------------
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        v_ = v.to(ema_state[k].device, non_blocking=True)
                        ema_state[k].mul_(ema_decay).add_(v_, alpha=1 - ema_decay)

                train_loss += loss.item()

            # --------- валидация -----------------------------------------
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _, _ = model(xb)
                    val_loss += smooth_bce_loss(logits, yb.float().unsqueeze(1)).item()
            val_loss /= len(val_loader)

            # --------- логирование ---------------------------------------
            train_loss /= len(train_loader)
            history["train"].append(train_loss)
            history["val"].append(val_loss)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss",   val_loss,   step=epoch)

            print(f"[{epoch+1}/{num_epochs}] train {train_loss:.4f} | val {val_loss:.4f}")

            # --- SWA после N эпох ----------------------------------------
            if epoch + 1 >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            # --- сохранить лучший чек-пойнт --------------------------------
            if val_loss < best_val_loss:
                best_val_loss, best_state = val_loss, {k: v.cpu() for k, v in ema_state.items()}

        # --------------- финализируем SWA-модель ----------------------------
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        best_state = swa_model.module.state_dict()  # окончательный стейт

        # --------------- сохраняем -----------------------------------------
        model_path = paths["models_dir"] / "mish_best_final_model_new_vers.pt"
        torch.save(best_state, model_path)
        mlflow.log_artifact(str(model_path))
        print(f"✔ Saved best model with val_loss={best_val_loss:.4f}")

    return best_state



def stacking():
    paths = get_project_paths()
    # --- Загружаем предсказания и метки -------------------------------------------
    pred1 = np.load(paths["vectors_dir"] /"pred_tanh_drop_AdamW_wd0.npy")
    pred2 = np.load(paths["vectors_dir"] /"pred_mish_Adam_wd0.npy")
    pred3 = np.load(paths["vectors_dir"] /"pred_mish_best_final_model_v2.npy")
    y = np.load(paths["vectors_dir"] /"labels_mish_best_final_model_v2.npy")

    # Проверка размеров
    assert pred1.shape == pred2.shape == pred3.shape == y.shape

    # Переводим логиты в вероятности
    def to_prob(x):
        if (x.min() < 0) and (x.max() > 1):
            return 1 / (1 + np.exp(-x))
        return x

    X = np.vstack([to_prob(pred1), to_prob(pred2), to_prob(pred3)]).T  # (N, 3)

    # --- Обучение LightGBM-классификатора -----------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_f1 = 0
    best_model = None
    best_thr = 0.5

    for train_idx, valid_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[valid_idx]
        y_train, y_val = y[train_idx], y[valid_idx]

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]

        # Подбор оптимального порога по F1
        for thr in np.linspace(0.05, 0.95, 19):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(y_val, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_model = model

    # --- Финальное предсказание и метрики -----------------------------------------
    stack_prob = best_model.predict_proba(X)[:, 1]
    stack_pred = (stack_prob >= best_thr).astype(int)

    acc = accuracy_score(y, stack_pred)
    f1 = f1_score(y, stack_pred)
    auc = roc_auc_score(y, stack_prob)
    cm = confusion_matrix(y, stack_pred)
    log_confusion_matrix(y, stack_pred, "meta_model")

    print(f"LightGBM STACK  ACC={acc:.4f} | F1={f1:.4f} | AUC={auc:.4f} | thr={best_thr:.2f}")
    print("Confusion matrix:\n", cm)

    # --- Сохраняем модель и порог -------------------------------------------------
    joblib.dump({"meta": best_model, "threshold": best_thr}, paths['models_dir']/"lightgbm_stack.pkl")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()
    evaluate_best_model()

    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
