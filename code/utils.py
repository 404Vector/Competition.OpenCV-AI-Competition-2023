import os
from dataset import MyDataset
from typing import Tuple
from torch.utils.data import random_split, Dataset
from converter import InputConverter, LabelConverter
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn


def set_random_seed(seed) -> None:
    # 파이토치의 랜덤시드 고정
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu 1개 이상일 때

    # 넘파이 랜덤시드 고정
    np.random.seed(seed)

    # CuDNN 랜덤시드 고정
    cudnn.benchmark = False
    cudnn.deterministic = True  # 연산 처리 속도가 줄어들어서 연구 후반기에 사용하자

    # 파이썬 랜덤시드 고정
    random.seed(seed)


def get_project_path() -> str:
    return os.path.dirname(os.path.dirname(__file__))


def split_dataset_by_ratio(
    dataset: MyDataset, split_ratio: float
) -> Tuple[Dataset, Dataset]:
    assert split_ratio > 0 and split_ratio < 1
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = dataset_size - train_size
    train_dataset, validation_dataset = random_split(
        dataset, [train_size, validation_size]
    )
    return train_dataset, validation_dataset


def inject_converter(
    dataset: Dataset,
    input_converter: InputConverter,
    label_converter: LabelConverter,
) -> None:
    dataset.dataset.input_transform = input_converter  # type: ignore
    dataset.dataset.label_transform = label_converter  # type: ignore
