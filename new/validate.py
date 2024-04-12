# python new/train.py
# val_loss           0.0001406194205628708

from model import AutoEncoder_LSTM, AutoEncoder_Wrapper
import torch

import os
import pandas as pd
from sklearn.model_selection import train_test_split

data_path = 'ready_for_training'
model_path = 'mouse_movement_anomaly_detection_model.pth'


def load_and_combine_csv(directory):
    combined_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)
    return combined_data


def eval_model(training_dir, sequence_length, test_size):
    combined_data = load_and_combine_csv(training_dir)
    flattened_sequences = combined_data.values

    n_features = flattened_sequences.shape[1]

    num_sequences = len(flattened_sequences) // sequence_length
    sequences = flattened_sequences[:num_sequences * sequence_length].reshape(
        (num_sequences, sequence_length, n_features))

    model = AutoEncoder_LSTM(n_features)
    model_wrapper = AutoEncoder_Wrapper(model)
    model_wrapper.load_model(model_path)
    model_wrapper.requires_grad_(False)
    if torch.cuda.is_available():
        model.cuda()
    val_loader = model_wrapper.get_dataloader(sequences)
    trainer = model_wrapper.get_trainer(1, devices=1)
    trainer.validate(model=model_wrapper, dataloaders=val_loader)


eval_model(data_path, 10, 0.2)
