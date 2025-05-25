import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from torchvision import transforms
import seaborn as sns
import mlflow
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import io, random



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

    paths = get_project_paths()

    # Вычисление матрицы ошибок
    labels = sorted(np.unique(all_labels))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    # Визуализация
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Сохранение и логирование
    plot_path = paths["image_dir"] / f"{model_name_tag}_confusion_matrix.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
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




def plot_training_results():
    activation_keywords = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "swish", "mish"]
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    with open(PROJECT_ROOT / "params.yaml") as f:
        paths = get_project_paths()

    image_dir = paths["image_dir"]
    metrics_dir = paths["models_dir"]

    # 1. Собираем изображения
    all_images = list(image_dir.glob("*.png"))


    images = [img for img in all_images if "confusion_matrix" not in img.name]

    # 3. Группировка по активационной функции
    groups = defaultdict(list)
    for img in images:
        for act in activation_keywords:
            if act in img.name:
                groups[act].append(img)
                break
        else:
            groups["unknown"].append(img)

    # 4. Сортировка внутри каждой группы по длине имени
    for key in groups:
        groups[key].sort(key=lambda x: len(x.name))

    # 5. Отображение
    for act_func, imgs in groups.items():
        print(f"\n=== Activation: {act_func.upper()} ===\n")
        n = len(imgs)
        rows = (n + 2) // 3  # по 3 в строку
        fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows), squeeze=False)
        axes = axes.flatten()

        for ax in axes[len(imgs):]:
            ax.axis('off')

        for ax, img_path in zip(axes, imgs):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path.name, fontsize=10)

            # Пытаемся найти соответствующий JSON
            base_name = img_path.stem.replace("training_loss_curve_", "")
            json_name = f"{base_name}_train_metrics.json"  # единственное подчёркивание
            json_path = metrics_dir / json_name

            if json_path.exists():
                with open(json_path) as f:
                    metrics = json.load(f)
                info = (f"Best val: {metrics.get('best_val_loss', 'NA'):.4f}\n"
                        f"Train: {metrics.get('final_train_loss', 'NA'):.4f}\n"
                        f"Val: {metrics.get('final_val_loss', 'NA'):.4f}")
            else:
                info = "No metrics"

            ax.text(0.5, -0.15, info, ha="center", va="top",
                    fontsize=9, transform=ax.transAxes)


        plt.tight_layout()
        plt.show()

def show_feature_top_images(
    fc_path: Path,
    parquet_path: Path,
    num_features: int = 5,  # сколько нейронов взять (строк)
    top_k: int = 6,         # сколько картинок на нейрон (колонок)
    random_state: int = None,
    thumb_size: int = 96,   # размер миниатюр
):
    """
    fc_path      – .npy с признаками формы (N, 1000)
    parquet_path – .parquet с колонкой image.bytes
    """
    # reproducible randomness (или каждый раз новый порядок, если оставить None)
    rnd = random.Random(random_state)

    # --- загрузка данных ----------------------------------------------------
    X = np.load(fc_path)              # (N, 1000)
    df = pd.read_parquet(parquet_path)
    assert len(X) == len(df), "Несовпадение размеров!"

    N, D = X.shape
    feat_idx = rnd.sample(range(D), k=min(num_features, D))     # случайные нейроны

    # --- рисуем сетку --------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=len(feat_idx), ncols=top_k, figsize=(top_k*2, num_features*2.2)
    )
    if len(feat_idx) == 1:
        axes = np.expand_dims(axes, 0)  # унифицируем доступ

    for row, f in enumerate(feat_idx):
        # берём top-k индексов по убыванию активации данного нейрона
        top_indices = np.argsort(-X[:, f])[:top_k]

        for col, img_idx in enumerate(top_indices):
            ax = axes[row, col]
            # извлекаем bytes → PIL → RGB → thumbnail
            img_bytes = df.iloc[img_idx]['image']['bytes']
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img.thumbnail((thumb_size, thumb_size))

            ax.imshow(img)
            ax.axis("off")
            if col == 0:                       # подписать строку слева
                ax.set_ylabel(f"f{f}", rotation=0, labelpad=30, fontsize=9)

    plt.suptitle(
        f"Топ-{top_k} картинок\nдля {num_features} случайно выбранных нейронов fc1000",
        y=0.92, fontsize=12
    )
    plt.tight_layout()
    plt.show()



def show_random_fc_activations(
    fc_path: Path,
    parquet_path: Path,
    neuron_indices=None,   # список конкретных фич; если None — выберем случайно
    num_features: int = 10, # сколько фич показать, если neuron_indices=None
    index: int = None,     # какой пример взять; если None — случайный
    random_state: int = None
):
    """
    fc_path      – .npy (N, 1000) с признаками
    parquet_path – .parquet с колонкой image.bytes
    """
    rnd = random.Random(random_state)

    # --- загрузка ---
    X  = np.load(fc_path)                # (N, 1000) float32/64
    df = pd.read_parquet(parquet_path)   # должен содержать column 'image'

    assert len(X) == len(df), "Несовпадение размеров fc_vector и parquet!"

    N, D = X.shape
    if index is None:
        index = rnd.randrange(N)         # случайный пример

    img_bytes = df.iloc[index]['image']['bytes']
    fc_vector = X[index]                 # (1000,)

    # --- какие нейроны показать ---
    if neuron_indices is None:
        neuron_indices = rnd.sample(range(D), k=min(num_features, D))
    else:
        # защита: обрезаем индексы, которые > 999
        neuron_indices = [i for i in neuron_indices if 0 <= i < D]
        if len(neuron_indices) == 0:
            raise ValueError("Список neuron_indices пуст или вне диапазона 0–999")

    # --- визуализация ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # оригинал
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    axs[0].imshow(img)
    axs[0].axis("off")
    axs[0].set_title(f"Image #{index}")

    # bar-chart
    values = fc_vector[neuron_indices]
    axs[1].bar(range(len(neuron_indices)), values)
    axs[1].set_xticks(range(len(neuron_indices)))
    axs[1].set_xticklabels([f"f{idx}" for idx in neuron_indices])
    axs[1].set_title("fc1000 activations")

    plt.tight_layout()
    plt.show()

# def show_activation_maps(conv3_out, max_channels=16):
#     """
#     conv3_out: torch.Tensor (C, H, W) или (B, C, H, W)
#     Показываем max_channels первых каналов из активаций.
#     """
#     paths = get_project_paths()
#     parquet_path = paths["raw_dir"] / "test.parquet",
#     if conv3_out.dim() == 4:
#         conv3_out = conv3_out[0]  # берем первый пример в батче (C, H, W)
#
#     n_channels = min(conv3_out.shape[0], max_channels)
#     fig, axes = plt.subplots(1, n_channels, figsize=(n_channels*2, 2))
#
#     for i in range(n_channels):
#         ax = axes[i] if n_channels > 1 else axes
#         act_map = conv3_out[i].cpu().numpy()
#         ax.imshow(act_map, cmap='viridis')
#         ax.axis('off')
#         ax.set_title(f"Ch {i}")
#
#     plt.show()
def show_activation_maps(
    conv3_out: torch.Tensor,
    idx: int = None,          # какой пример взять из parquet; если None, нужен img_bytes
    img_bytes: bytes = None,  # можно сразу передать
    max_channels: int = 16
):
    """
    conv3_out — Tensor (C,H,W) или (B,C,H,W) c активациями этого же изображения.
    idx       — индекс картинки в test.parquet (приоритетнее, если img_bytes не задан).
    img_bytes — необязательные байты изображения (если нет parquet).
    """

    # --- достаем оригинал ----------------------------------------------------
    if img_bytes is None:
        if idx is None:
            raise ValueError("Нужно передать либо idx, либо img_bytes!")
        paths = get_project_paths()
        parquet_path = paths["raw_dir"] / "test.parquet"
        df = pd.read_parquet(parquet_path)
        img_bytes = df.iloc[idx]["image"]["bytes"]

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # --- нормализуем conv3_out -> (C,H,W) ------------------------------------
    if conv3_out.dim() == 4:
        conv3_out = conv3_out[0]  # (C,H,W)

    n_channels = min(conv3_out.shape[0], max_channels)

    # --- рисуем --------------------------------------------------------------
    fig_cols = n_channels + 1
    fig, axes = plt.subplots(1, fig_cols, figsize=(2*fig_cols, 2.5))

    # оригинал
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # карты активаций
    for i in range(n_channels):
        ax = axes[i+1]
        fmap = conv3_out[i].cpu().numpy()
        ax.imshow(fmap, cmap="viridis")
        ax.set_title(f"Ch {i}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()



def grad_cam(img_bgr, class_idx=None, thr=0.4):
    from nn_train import MyCNN
    import cv2

    paths = get_project_paths()
    model_path = paths["models_dir"] / "mish_best_final_model_new_vers.pt"
    activation = "mish"

    dropout = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyCNN(activation=activation, dropout=dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))  # 👈 обязательно!
    tensor = transforms.ToTensor()(img_resized).unsqueeze(0).to(device)

    # Прямой проход: logits, features, conv3_out
    logits, features, conv3_out = model(tensor)

    if class_idx is None:
        class_idx = (logits > 0).int().item()  # 0 или 1, твоя задача бинарная

    score = logits[0, 0]

    model.zero_grad()
    score.backward(retain_graph=True)

    # Получаем градиенты conv3_out (B, C, H, W)
    gradients = model.gradients.cpu().data.numpy()[0]  # (C, H, W)
    activations = conv3_out.cpu().data.numpy()[0]      # (C, H, W)

    # Усреднение градиентов по пространственным осям — веса для каналов
    weights = np.mean(gradients, axis=(1, 2))          # (C,)

    # Grad-CAM = взвешенное суммирование активаций
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
    for i, w in enumerate(weights):
        cam += w * activations[i]

    # Релу на карте
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()  # нормализация 0..1

    # Приводим cam к размеру исходного изображения
    cam = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))

    # Наложение тепловой карты на изображение
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = heatmap * 0.4 + img_rgb * 0.6
    overlayed = np.uint8(overlayed)

    # Возвращаем предсказанный класс, raw score и картинку с CAM
    return class_idx, score.item(), overlayed, cam


def show_grad_cam_results(original_img, cam_overlayed, cam_mask):
    """
    original_img: RGB изображение (np.array)
    cam_overlayed: RGB изображение с наложенной тепловой картой
    cam_mask: np.array с float32, значения от 0 до 1
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(cam_overlayed)
    axs[1].set_title("Grad-CAM Overlay")
    axs[1].axis("off")

    im = axs[2].imshow(cam_mask, cmap='jet')
    axs[2].set_title("Grad-CAM Mask")
    axs[2].axis("off")
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()