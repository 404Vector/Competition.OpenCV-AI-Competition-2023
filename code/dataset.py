import os
from typing import Any, Optional
import cv2
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        target_extension: str = ".png",
        input_transform: Optional[Any] = None,
        label_transform: Optional[Any] = None,
    ):
        dataset_path = os.path.abspath(dataset_path)
        assert os.path.exists(dataset_path), f"ERROR, [{dataset_path}] is not exist."
        self._directory_path = dataset_path

        _b_path = [
            os.path.join(dataset_path, "B", _f)
            for _f in os.listdir(os.path.join(self._directory_path, "B"))
            if _f.endswith(target_extension)
        ]
        _g_path = [
            os.path.join(dataset_path, "G", _f)
            for _f in os.listdir(os.path.join(self._directory_path, "G"))
            if _f.endswith(target_extension)
        ]
        _r_path = [
            os.path.join(dataset_path, "R", _f)
            for _f in os.listdir(os.path.join(self._directory_path, "R"))
            if _f.endswith(target_extension)
        ]

        self._file_paths = _b_path + _g_path + _r_path
        self.input_transform = input_transform
        self.label_transform = label_transform

    @property
    def file_paths(self):
        return self._file_paths

    def __len__(self):
        return len(self._file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        path_elements = path.split("/")
        category = path_elements[-2]
        rgb = [
            int(_v) for _v in path_elements[-1].split("_")[-1].split(".")[0].split(",")
        ]
        label = (category, *rgb)
        if self.input_transform:
            image = self.input_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label
