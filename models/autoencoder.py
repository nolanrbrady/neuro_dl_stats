import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO - will be base class

class AutoEncoder(nn.Module):
    def __init__(self, data_shape, latent_dim=1):
        super(AutoEncoder, self).__init__()

        # Extract input dimension from data_shape
        self.input_dim = data_shape[0]

        # Encoder layers
        self.encoder_fc1 = nn.Linear(self.input_dim, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.latent_layer = nn.Linear(64, latent_dim)

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, self.input_dim)  # Output back to input_dim (4)
    
    # DRY - created encode/decode methods since these lines were repeated in forward and get_latent_value
    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        return self.latent_layer(x)

    def decode(self, z):
        x = F.relu(self.decoder_fc1(z))
        x = F.relu(self.decoder_fc2(x))
        return torch.sigmoid(self.decoder_fc3(x))  # Output between 0 and 1

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

    def get_latent_value(self, x):
        with torch.no_grad():
            return self.encode(x)

    def train(self, x, learning_rate=0.0001, n_epochs=100):
        # Convert input to PyTorch tensor
        x_train = torch.FloatTensor(x)
        print("X_train shape: ", x_train.shape)

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            recon_x = self(x_train)

            # Compute loss
            loss = F.mse_loss(recon_x, x_train)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')

        return self