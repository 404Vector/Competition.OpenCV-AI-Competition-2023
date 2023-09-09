import os
from utils import (
    get_project_path,
    split_dataset_by_ratio,
    inject_converter,
    set_random_seed,
)
from tqdm import tqdm
from dataset import MyDataset
from argparse import ArgumentParser, Namespace
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from model import ColorRegressor
from converter import InputConverter, LabelConverter


def parse_arges() -> Namespace:
    paser = ArgumentParser()

    return paser.parse_args(())


def train_model(model, train_dataloader, criterion, optimizer, device, epoch_index):
    cost = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cost += loss
    cost = cost / len(train_dataloader)
    return cost


def valid_model(model, valid_dataloader, criterion, device, epoch_index):
    cost = 0
    with torch.no_grad():
        model.eval()
        for images, labels in valid_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            cost += loss
        cost = cost / len(valid_dataloader)
    return cost


def main(args: Namespace):
    assert torch.cuda.is_available(), "ERROR::CUDA not available."

    # set param
    device = "cuda"
    split_ratio = 0.8
    epoch = 50
    lr = 1e-4
    batch_size = 16
    random_seed = 42

    # set random seed
    set_random_seed(random_seed)

    # create instance
    project_path = get_project_path()
    dataset_path = os.path.join(project_path, "resource")
    dataset = MyDataset(dataset_path=dataset_path)
    model = ColorRegressor().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataset, validation_dataset = split_dataset_by_ratio(
        dataset=dataset, split_ratio=split_ratio
    )
    inject_converter(
        dataset=train_dataset,
        input_converter=InputConverter("train"),
        label_converter=LabelConverter(),
    )
    inject_converter(
        dataset=validation_dataset,
        input_converter=InputConverter("test"),
        label_converter=LabelConverter(),
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # train
    process_bar = tqdm(range(epoch))
    for epoch_index in process_bar:
        train_cost = train_model(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch_index=epoch_index,
        )
        valid_cost = valid_model(
            model=model,
            valid_dataloader=validation_dataloader,
            criterion=criterion,
            device=device,
            epoch_index=epoch_index,
        )
        process_bar.set_description(
            f"epoch {epoch_index:#04d}| train_cost {train_cost:#.6f}| valid_cost {valid_cost:#.6f}|"
        )


if __name__ == "__main__":
    args = parse_arges()
    main(args)
