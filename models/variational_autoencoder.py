import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO - make variational autoencoder extend autoencoder

class VariationalAutoEncoder(nn.Module):
    def __init__(self, data_shape, latent_dim=1):
        super(VariationalAutoEncoder, self).__init__()

        # Extract input dimension from data_shape
        self.input_dim = data_shape[0]  # This will be 4

        # Encoder layers
        self.encoder_fc1 = nn.Linear(self.input_dim, 128)
        self.encoder_fc2 = nn.Linear(128, 64)
        self.mu_layer = nn.Linear(64, latent_dim)       # Mean of latent space
        self.logvar_layer = nn.Linear(64, latent_dim)  # Log variance of latent space

        # Decoder layers
        self.decoder_fc1 = nn.Linear(latent_dim, 64)
        self.decoder_fc2 = nn.Linear(64, 128)
        self.decoder_fc3 = nn.Linear(128, self.input_dim)  # Output back to input_dim (4)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encoding
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)

        # Latent space sampling
        z = self.reparameterize(mu, logvar)

        # Decoding
        x = F.relu(self.decoder_fc1(z))
        x = F.relu(self.decoder_fc2(x))
        x = torch.sigmoid(self.decoder_fc3(x))  # Output between 0 and 1
        return x, mu, logvar

    def get_latent_value(self, x):
        with torch.no_grad():
            x = F.relu(self.encoder_fc1(x))
            x = F.relu(self.encoder_fc2(x))
            mu = self.mu_layer(x)
            logvar = self.logvar_layer(x)
            z = self.reparameterize(mu, logvar)
        return z

    def train(self, x, learning_rate=0.0001, n_epochs=100):
        # Convert input to PyTorch tensor
        x_train = torch.FloatTensor(x)

        # Define loss function
        def vae_loss(recon_x, x, mu, logvar):
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss + kld

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(n_epochs):
            # Forward pass
            recon_x, mu, logvar = self(x_train)

            # Compute loss
            loss = vae_loss(recon_x, x_train, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')

        return self