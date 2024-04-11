

# Sequential([
#         LSTM(64, activation='tanh', return_sequences=True),
#         Dropout(0.2),  # Added dropout for regularization
#         LSTM(128, activation='tanh', return_sequences=True),
#         TimeDistributed(Dense(n_features, activation='linear'))
#     ])

import pytorch_lightning as pl
import torch
import numpy as np

import torch.utils.data


class AutoEncoder_LSTM(torch.nn.Module):
    def __init__(self, input_feats: int, **kwargs):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout(0.2)
        # Encoder
        self.lstm1 = torch.nn.LSTM(
            input_size=input_feats, hidden_size=128, batch_first=True)
        self.lstm2 = torch.nn.LSTM(
            input_size=128, hidden_size=64, batch_first=True)
        # Decoder
        self.lstm3 = torch.nn.LSTM(
            input_size=64, hidden_size=64, batch_first=True)
        self.lstm4 = torch.nn.LSTM(
            input_size=64, hidden_size=128, batch_first=True)

        self.fc = torch.nn.Linear(128, input_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor
        # Encoder
        # (batch, timesteps, feature)
        out, _ = self.lstm1(x)
        out = self.tanh(out)
        out = self.dropout(out)
        # (batch, timesteps, 128)
        out, _ = self.lstm2(out)
        out = self.tanh(out)
        # (batch, timesteps, 64)

        out = out[:, -1, None].repeat(1, x.size(1), 1)
        # (batch, timesteps, 64)

        # Decoder
        out, _ = self.lstm3(out)
        out = self.tanh(out)
        out = self.dropout(out)
        # (batch, timesteps, 64)
        out, _ = self.lstm4(out)
        out = self.tanh(out)
        # (batch, timesteps, 128)
        out = self.fc(out)
        return out


class AutoEncoder_Wrapper(pl.LightningModule):
    def __init__(self, model: AutoEncoder_LSTM):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss_fn(self(x), x)  # 使用重建损失作为训练损失
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]
        loss = self.loss_fn(self(x), x)  # 使用重建损失作为验证损失
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_trainer(self, num_epochs: int):
        return pl.Trainer(
            max_epochs=num_epochs,
            callbacks=[pl.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, mode='min')]
        )

    def get_dataloader(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        batch_size: int = 32,
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        return (
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X_train).float()),
                batch_size=batch_size),
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.from_numpy(X_test).float()),
                batch_size=batch_size),
        )

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)
