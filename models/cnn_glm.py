import torch
from torch import nn
import torch.optim as optim

class SeqCNN(nn.Module):
    def __init__(self, data):
        super(SeqCNN, self).__init__()
        in_channels = 1
        out_channels = 1
        kernel_size = data.shape[1]
        num_layers = 1
        hidden_size = data.shape[1]
        self.betas = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        nn.init.xavier_uniform_(self.betas)

        self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        self.rnn = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels)  # Adjust 10 to your desired output size

        # Use the kernal in the CNN layer
        with torch.no_grad():
            self.cnn.weight = self.betas

    def get_betas(self):
        return self.betas

    def forward(self, x):
        # print(x.shape)
        x = x.unsqueeze(1)
        # print(x.shape)
        # CNN feature extraction
        x = self.cnn(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)

        # Reshape for RNN input
        x = x.view(x.size(0), x.size(1), -1)

        # RNN processing
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Take the last hidden state

        # Another relu layer
        x = torch.relu(x)

        # Final fully connected layer
        x = self.fc(x)
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