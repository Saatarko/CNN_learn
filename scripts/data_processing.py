import argparse
import os

import mlflow
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from nn_train import ByteImageDataset
from utils import  get_project_paths
from task_registry import main, task
from datasets import load_dataset

mlflow.set_tracking_uri('http://localhost:5000')


@task("data:split_and_save_dataset")
def split_and_save_dataset():
    dataset = load_dataset("microsoft/cats_vs_dogs", split="train")
    paths = get_project_paths()

    train_test = dataset.train_test_split(test_size=0.2, seed=42)
    val_train = train_test["train"].train_test_split(test_size=0.3, seed=42)

    train_dataset = val_train["train"]
    val_dataset = val_train["test"]
    test_dataset = train_test["test"]


    # Сохраняем в формате parquet
    train_dataset.to_parquet(paths["raw_dir"]/"train.parquet")
    val_dataset.to_parquet(paths["raw_dir"]/"val.parquet")
    test_dataset.to_parquet(paths["raw_dir"]/"test.parquet")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", help="Список задач для выполнения")
    args = parser.parse_args()


    if args.tasks:
        main(args.tasks)  # Здесь передаем задачи, которые указаны в командной строке
