from typing import List, Tuple
import torch as th
import numpy as np
from torch.utils.data import  DataLoader
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: th.optim.Optimizer,
        device: th.device,
        n_epochs: int = 100,
        verbose: bool = True
) -> Tuple[float, float]:
    """Train model and return final train/test losses"""
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_losses = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Testing
        model.eval()
        test_losses = []
        with th.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                test_losses.append(loss.item())

        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}: Train={np.mean(train_losses):.6f}, '
                  f'Test={np.mean(test_losses):.6f}')

    return np.mean(train_losses), np.mean(test_losses)