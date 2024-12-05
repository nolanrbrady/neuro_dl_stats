import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .autoencoder import AutoEncoder

class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, data_shape, latent_dim=1):
        super(VariationalAutoEncoder, self).__init__(data_shape, latent_dim)

        # Override latent layer with mean and log variance layers
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

    # new method, DRY 
    def encode(self, x):
        x = F.relu(self.encoder_fc1(x))
        x = F.relu(self.encoder_fc2(x))
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_latent_value(self, x):
        with torch.no_grad():
            mu, logvar = self.encode(x)
            return self.reparameterize(mu, logvar)


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