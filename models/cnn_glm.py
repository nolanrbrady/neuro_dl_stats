import torch
import torch.nn as nn
import torch.optim as optim


class SeqCNN(nn.Module):
    def __init__(self, data_shape, out_channels=1, kernel_size=None):
        super(SeqCNN, self).__init__()
        print("data_shape: ", data_shape)
        in_channels = 1
        self.out_channels = out_channels
        self.kernel_size = kernel_size or data_shape.shape[1]
        self.hidden_size = data_shape.shape[1]

        # Initialize parameters
        self.betas = nn.Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size))
        nn.init.xavier_uniform_(self.betas)

        self.cnn = nn.Conv1d(in_channels, out_channels, self.kernel_size, stride=1, padding=0)
        self.rnn = nn.LSTM(out_channels, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, out_channels)

        # Use the kernel in the CNN layer
        with torch.no_grad():
            self.cnn.weight = self.betas

    def get_betas(self):
        return self.betas

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = torch.relu(x)
        x = self.fc(x)
        return x

    def train_model(self, x, y, learning_rate=0.05, n_epochs=20000):
        x_train = torch.FloatTensor(x)
        y = torch.FloatTensor(y).view(-1)
        model = self
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(n_epochs):
            y_pred = model(x_train).view(-1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return model, losses