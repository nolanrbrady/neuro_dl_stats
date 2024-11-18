import torch
from torch import nn
import torch.optim as optim

class AutoEncoder(nn.Module):
    def __init__(self, data):
        super(AutoEncoder, self).__init__()

    def get_betas(self):
        return self.betas

    def forward(self, x):
        print("NEED TO IMPLEMENT THIS")
        return x

    def train(x, y, learning_rate=0.05, n_epochs=20000):
        print(x.shape, y.shape)
        # Convert inputs to PyTorch tensors
        x_train = torch.FloatTensor(x)
        y = torch.FloatTensor(y).view(-1)
        
        # Initialize model
        model = SeqCNN(x_train)
        
        # Define loss function
        criterion = nn.MSELoss()
        
        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training history
        losses = []
        patience = 10
        
        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            y_pred = model(x_train).view(-1)
            
            # Compute loss
            loss = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store loss
            losses.append(loss.item())
            
            # print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}')

                
        return model, losses