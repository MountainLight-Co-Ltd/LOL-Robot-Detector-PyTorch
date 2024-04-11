from model import AutoEncoder_LSTM, AutoEncoder_Wrapper

import os
import pandas as pd
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


def train_model(training_dir, sequence_length, test_size):
    combined_data = load_and_combine_csv(training_dir)
    flattened_sequences = combined_data.values

    n_features = flattened_sequences.shape[1]

    num_sequences = len(flattened_sequences) // sequence_length
    sequences = flattened_sequences[:num_sequences * sequence_length].reshape(
        (num_sequences, sequence_length, n_features))

    X_train, X_test = train_test_split(
        sequences, test_size=test_size, random_state=42)
    model = AutoEncoder_LSTM(n_features)
    model_wrapper = AutoEncoder_Wrapper(model)
    train_loader, val_loader = model_wrapper.get_dataloader(X_train, X_test)
    trainer = model_wrapper.get_trainer(500)
    trainer.fit(model_wrapper, train_loader, val_loader)

    plt.plot(trainer.logged_metrics['train_loss'], label='Training Loss')
    plt.plot(trainer.logged_metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    model_wrapper.save('mouse_movement_anomaly_detection_model.pth')


train_model(data_path, 10, 0.2)

print("Model training complete and saved.")
