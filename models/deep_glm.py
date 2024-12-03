import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim


class BoldGLM(nn.Module):
    def __init__(self, data_shape):
        super(BoldGLM, self).__init__()
        self.n_predictors = data_shape.shape[1]
        self.data_shape = data_shape.shape
        # Create learnable beta parameters with size matching the number of predictors
        self.betas = nn.Parameter(torch.randn(self.n_predictors, 1))  # Initialize with random values

    def forward(self, X):
        # Perform matrix multiplication (dot product between X and betas)
        return X @ self.betas

    def get_betas(self):
        return self.betas.detach().numpy()

    @staticmethod
    def _check_convergence(losses, current_loss, patience):
        if len(losses) < patience:
            return False
        return all(x <= current_loss for x in losses[-patience:])

    def train_model(self, X, y, learning_rate=0.05, n_epochs=20000, patience=10):
        # Convert inputs to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).view(-1)

        # Define loss function
        criterion = nn.MSELoss()

        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training history
        losses = []

        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            y_pred = self(X).view(-1)

            # Compute loss
            loss = criterion(y_pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store loss
            losses.append(loss.item())

            # Print progress (optional)
            if epoch % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")

            # Early stopping based on patience
            if self._check_convergence(losses, loss.item(), patience):
                print(f"Converged at epoch {epoch + 1} with loss: {loss.item():.4f}")
                break

        return self, losses