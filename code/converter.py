import numpy as np
from numpy import ndarray
from typing import Literal
from torch import Tensor
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class InputConverter:
    def __init__(self, mode: Literal["train", "test"]) -> None:
        self.mode = mode
        self.train_input_tf = A.Compose(
            [
                A.VerticalFlip(),
                A.HorizontalFlip(),
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        self.test_input_tf = A.Compose(
            [A.Resize(224, 224), A.Normalize(), ToTensorV2()]
        )

    def __call__(self, input_data: ndarray) -> Tensor:
        if len(input_data.shape) == 3:
            input_image = np.expand_dims(input_data, axis=0)
        tf = self.train_input_tf if self.mode == "train" else self.test_input_tf
        tf_result = tf(image=input_data)["image"]
        return tf_result


class LabelConverter:
    def __init__(self) -> None:
        self.category_map = {
            "R": 0,
            "G": 1,
            "B": 2,
        }

    def __call__(self, label_data):
        category, r, g, b = label_data
        category = self.category_map[category]
        return np.array([r, g, b], np.float32) / 255.0
