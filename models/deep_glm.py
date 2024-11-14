import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

class BoldGLM(nn.Module):
    def __init__(self, n_predictors):
        super(BoldGLM, self).__init__()
        # Create learnable beta parameters with size matching the number of predictors
        self.betas = nn.Parameter(torch.randn(n_predictors, 1))  # Initialize with random values
    
    def forward(self, X):
        # X should have shape (batch_size, n_predictors)
        # Perform matrix multiplication (dot product between X and betas)
        # print("Self.betas", self.betas)
        return X @ self.betas
    
    def get_betas(self):
        return self.betas.detach().numpy()
        
    def check(losses, current_loss, patience):
        if (len(losses) < 5): 
            return False
    
        list1 = losses[-patience:]
        return all(x <= current_loss for x in list1)
    
    def train_bold_glm(X, y, learning_rate=0.05, n_epochs=20000):
        print(X.shape, y.shape)
        # Convert inputs to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).view(-1)
        
        # Initialize model
        model = BoldGLM(n_predictors=X.shape[1])
        
        # Define loss function
        criterion = nn.MSELoss()
        
        # Define optimizer
        # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        losses = []
        patience = 10
        
        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            y_pred = model(X).view(-1)
            
            # Compute loss
            loss = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            
            # print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')
            
            # if (check(losses, loss, patience)):
            #     break;
                
        return model, losses