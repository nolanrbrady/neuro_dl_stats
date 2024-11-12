import pandas as pd
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

class SeqGLM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_size, num_layers):
        super(RCNN, self).__init__()

        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.rnn = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 10)  # Adjust 10 to your desired output size

    def forward(self, x):
        # CNN feature extraction
        x = self.cnn(x)
        x = torch.relu(x)

        # Reshape for RNN input
        x = x.view(x.size(0), x.size(1), -1)

        # RNN processing
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last hidden state

        # Final fully connected layer
        x = self.fc(x)
        return x

    def train(X, y, learning_rate=0.05, n_epochs=20000):
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