import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_path = 'ready_for_training'

def load_and_combine_csv(directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

def train_model(training_dir, sequence_length, test_size):
    combined_data = load_and_combine_csv(training_dir)
    flattened_sequences = combined_data.values

    n_features = flattened_sequences.shape[1]

    num_sequences = len(flattened_sequences) // sequence_length
    sequences = flattened_sequences[:num_sequences * sequence_length].reshape((num_sequences, sequence_length, n_features))

    X_train, X_test = train_test_split(sequences, test_size=test_size, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float)
    X_test = torch.tensor(X_test, dtype=torch.float)

    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, X_test), batch_size=64, shuffle=False)

    class LSTM_Autoencoder(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super(LSTM_Autoencoder, self).__init__()
            self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers, batch_first=True)
        
        def forward(self, x):
            _, (hidden, cell) = self.encoder(x)
            outputs, _ = self.decoder(x, (hidden, cell))
            return outputs


    model = LSTM_Autoencoder(input_size=n_features, hidden_size=64, num_layers=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    num_epochs = 500
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(test_loader)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'mouse_movement_anomaly_detection_model.pt')

train_model(data_path, 10, 0.2)

print("Model training complete and saved.")
