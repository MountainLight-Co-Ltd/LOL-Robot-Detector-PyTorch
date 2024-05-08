import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_path = 'ready_for_training'

def load_and_combine_csv(directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data

class LSTMModel(nn.Module):
    def __init__(self, sequence_length, n_features):
        super(LSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.lstm1 = nn.LSTM(n_features, 128, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm3 = nn.LSTM(64, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm4 = nn.LSTM(64, 128, batch_first=True)
        self.time_distributed = nn.Linear(128, n_features)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :].unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.lstm3(x)
        x = self.dropout2(x)
        x, _ = self.lstm4(x)
        x = self.time_distributed(x)
        return x

def train_model(training_dir, sequence_length, test_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")

    combined_data = load_and_combine_csv(training_dir)
    flattened_sequences = combined_data.values

    n_features = flattened_sequences.shape[1]
    num_sequences = len(flattened_sequences) // sequence_length
    sequences = flattened_sequences[:num_sequences * sequence_length].reshape(
        (num_sequences, sequence_length, n_features))

    X_train, X_test = train_test_split(sequences, test_size=test_size, random_state=42)
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)

    train_data = TensorDataset(X_train, X_train)
    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    
    model = LSTMModel(sequence_length, n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

        if epoch % 10 == 0:  # check every 10 epochs
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_test)
                val_loss = criterion(val_predictions, X_test).item()
            print(f'Validation Loss after {epoch+1} epochs: {val_loss}')

    # Save the model
    torch.save(model.state_dict(), 'mouse_movement_anomaly_detection_model.pth')

train_model(data_path, 10, 0.2)

print("Model training complete and saved.")
