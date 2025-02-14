import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dtaidistance import dtw
import numpy as np
from tqdm import tqdm
import copy


def create_sequences(data_tensor, seq_length):
    sequences = []
    for i in range(len(data_tensor) - seq_length):
        seq = data_tensor[i : i + seq_length]
        sequences.append(seq)
    return torch.stack(sequences)


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_dims=[64, 32, 16]):
        super(Autoencoder, self).__init__()
        # Encoder
        encoder_layers = []
        curr_size = input_size
        for hd in hidden_dims:
            encoder_layers.append(nn.Linear(curr_size, hd))
            encoder_layers.append(nn.ReLU())
            curr_size = hd

        # Decoder
        decoder_layers = []
        hidden_dims.reverse()
        for hd in hidden_dims[1:]:
            decoder_layers.append(nn.Linear(curr_size, hd))
            decoder_layers.append(nn.ReLU())
            curr_size = hd

        decoder_layers.append(nn.Linear(curr_size, input_size))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch_size, seq_length*features) flatten
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(x.size(0), -1)  # reshape back
        return decoded


class CnnAutoencoder(nn.Module):
    def __init__(self, input_channels=1, sequence_length=100, hidden_dims=[16, 32]):
        super(CnnAutoencoder, self).__init__()

        assert len(hidden_dims) == 2, ""

        self.input_channels = input_channels
        self.sequence_length = sequence_length

        encoder_layers = []

        encoder_layers.append(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=hidden_dims[0],
                kernel_size=3,
                padding=1,
            )
        )
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        encoder_layers.append(
            nn.Conv1d(
                in_channels=hidden_dims[0],
                out_channels=hidden_dims[1],
                kernel_size=3,
                padding=1,
            )
        )
        encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.encoder = nn.Sequential(*encoder_layers)

        self.encoded_length = sequence_length // 2  # first MaxPool
        self.encoded_length = self.encoded_length // 2  # second MaxPool

        decoder_layers = []

        decoder_layers.append(
            nn.ConvTranspose1d(
                in_channels=hidden_dims[1],
                out_channels=hidden_dims[0],
                kernel_size=2,
                stride=2,
            )
        )
        decoder_layers.append(nn.ReLU())

        decoder_layers.append(
            nn.ConvTranspose1d(
                in_channels=hidden_dims[0],
                out_channels=input_channels,
                kernel_size=2,
                stride=2,
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if decoded.size(2) > x.size(2):
            decoded = decoded[:, :, : x.size(2)]
        elif decoded.size(2) < x.size(2):
            pad_amount = x.size(2) - decoded.size(2)
            decoded = nn.functional.pad(decoded, (0, pad_amount), "constant", 0.0)

        return decoded


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


class CNNModel(nn.Module):
    """
    CNN-based model for reconstruction-based anomaly detection.
    """

    def __init__(self, input_size, out_channels=16, kernel_size=3):
        super(CNNModel, self).__init__()
        # 1 feature, channel=1
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=out_channels, kernel_size=kernel_size, padding=1
        )
        self.relu = nn.ReLU()
        # Flatten -> fully connected -> reshape
        self.fc = nn.Linear(out_channels * input_size, input_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1)
        x = x.transpose(1, 2)  # shape: (batch_size, 1, seq_len)
        x = self.conv1(x)  # (batch_size, out_channels, seq_len)
        x = self.relu(x)
        x = x.reshape(x.size(0), -1)  # flatten to (batch_size, out_channels * seq_len)
        x = self.fc(x)  # (batch_size, input_size)
        return x.unsqueeze(-1)  # (batch_size, input_size, 1)


class StackVAEG(nn.Module):
    def __init__(self, input_size, hidden_dims=[64, 32], latent_dim=16):
        super(StackVAEG, self).__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        in_dim = self.input_size
        for hdim in self.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hdim))
            encoder_layers.append(nn.ReLU())
            in_dim = hdim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, self.latent_dim)

        # Decoder
        decoder_layers = []
        out_dim = self.hidden_dims[-1]
        decoder_layers.append(nn.Linear(self.latent_dim, out_dim))
        decoder_layers.append(nn.ReLU())
        for hdim in reversed(self.hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(out_dim, hdim))
            decoder_layers.append(nn.ReLU())
            out_dim = hdim
        decoder_layers.append(nn.Linear(out_dim, self.input_size))

        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        # Flatten
        x = x.view(x.size(0), -1)
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        z = self.reparameterize(mu, logvar)
        reconstructed = self.decoder(z)

        return reconstructed.view(x.size(0), -1), mu, logvar


class OmniAnomaly(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, latent_dim=16, num_layers=1):
        super(OmniAnomaly, self).__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # LSTM Encoder
        self.lstm_enc = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # LSTM Decoder
        self.lstm_dec = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        enc_out, _ = self.lstm_enc(x)
        mu = self.fc_mu(enc_out)  # (batch_size, seq_len, latent_dim)
        logvar = self.fc_logvar(enc_out)

        z = self.reparameterize(mu, logvar)  # (batch_size, seq_len, latent_dim)

        dec_out, _ = self.lstm_dec(z)
        reconstructed = self.fc_out(dec_out)  # (batch_size, seq_len, input_size)

        return reconstructed, mu, logvar


class EGADSModel:
    def __init__(self, window=30):
        self.window = window
        self.train_length_ = None

    def fit(self, data):
        data = data.flatten()
        self.train_length_ = len(data)

    def predict(self, data):
        data = data.flatten()
        full_series = pd.Series(data)

        roll_mean = full_series.rolling(self.window, min_periods=1).mean()
        roll_std = full_series.rolling(self.window, min_periods=1).std()

        roll_std = roll_std.fillna(1e-8)

        z_scores = np.abs(data - roll_mean) / roll_std

        anomaly_scores = z_scores.to_numpy()
        return anomaly_scores


class DTWModel:
    def __init__(self, window=30, n_references=10, seed=42):
        self.window = window
        self.n_references = n_references
        self.references = []
        self.seed = seed

    def fit(self, train_sequences):
        np.random.seed(self.seed)
        total_sequences = train_sequences.shape[0]
        selected_indices = np.random.choice(
            total_sequences, self.n_references, replace=False
        )
        self.references = [
            train_sequences[i].squeeze().numpy() for i in selected_indices
        ]
        print(f"DTWModel: Selected {self.n_references} reference sequences.")

    def predict(self, test_sequences):
        scores = []
        for seq in test_sequences:
            seq_np = seq.squeeze().numpy()
            distances = [dtw.distance(seq_np, ref) for ref in self.references]
            min_distance = min(distances)
            scores.append(min_distance)
        return np.array(scores)


class NaiveKANLayer(nn.Module):
    """
    A simple time-distributed linear transform (no Fourier features).
    Applies a linear transform to each time step in [batch_size, window_size, inputdim].
    """

    def __init__(self, inputdim, outdim, addbias=True):
        super(NaiveKANLayer, self).__init__()
        self.linear = nn.Linear(inputdim, outdim, bias=addbias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, window_size, inputdim]
        Returns:
            Tensor of shape [batch_size, window_size, outdim]
        """
        # We apply the linear transform to each time step:
        # A direct way is to reshape x -> [batch_size*window_size, inputdim]
        batch_size, window_size, inputdim = x.shape
        x_reshaped = x.view(batch_size * window_size, inputdim)

        out = self.linear(x_reshaped)  # shape: [batch_size * window_size, outdim]
        out = out.view(batch_size, window_size, -1)
        return out


class KAN(nn.Module):
    """
    A simplified KAN model without Fourier transforms.
    Stacks multiple NaiveKANLayer layers + BatchNorm1d + LeakyReLU + Dropout,
    then does a global average pooling over time, and outputs a final linear projection.
    """

    def __init__(
        self,
        in_feat,  # e.g., 1 (for a single feature)
        hidden_feat,  # hidden dimension
        out_feat,  # final output dimension (e.g., 1)
        num_layers=2,  # how many NaiveKANLayers to stack
        use_bias=True,
        dropout=0.3,
    ):
        super(KAN, self).__init__()
        self.num_layers = num_layers

        # 1) Linear input projection
        self.lin_in = nn.Linear(in_feat, hidden_feat, bias=use_bias)
        self.bn_in = nn.BatchNorm1d(hidden_feat)
        self.dropout = nn.Dropout(p=dropout)

        # 2) Stacked naive layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                NaiveKANLayer(hidden_feat, hidden_feat, addbias=use_bias)
            )
            self.bns.append(nn.BatchNorm1d(hidden_feat))

        # 3) Final linear output
        self.lin_out = nn.Linear(hidden_feat, out_feat, bias=use_bias)

    def forward(self, x):
        """
        x shape: [batch_size, window_size, in_feat]
        Returns: [batch_size] if out_feat=1, or [batch_size, out_feat]
        """
        batch_size, window_size, _ = x.size()

        # (A) Input projection + BN + LeakyReLU + Dropout
        x = self.lin_in(x)  # shape: [batch_size, window_size, hidden_feat]
        # BN expects [N, C], so flatten time + batch:
        x = self.bn_in(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        # (B) Stacked naive layers
        for layer, bn in zip(self.layers, self.bns):
            x = layer(x)  # shape: [batch_size, window_size, hidden_feat]
            # BN again on flattened dimension
            x = bn(x.view(-1, x.size(-1))).view(batch_size, window_size, -1)
            x = F.leaky_relu(x, negative_slope=0.1)
            x = self.dropout(x)

        # (C) Global average pooling over the window dimension
        x = x.mean(dim=1)  # shape: [batch_size, hidden_feat]

        # (D) Final linear -> e.g. shape [batch_size, out_feat]
        x = self.lin_out(x).squeeze()
        return x


class AnomalyDetector:
    def __init__(self, model_name, model_params, sequence_length=100, device="cpu"):
        self.model_name = model_name.lower()
        self.model_params = model_params
        self.sequence_length = sequence_length
        self.device = device

        if self.model_name == "egads":
            self.model = EGADSModel(**self.model_params)
        elif self.model_name == "dtw":
            self.model = DTWModel(**self.model_params)
        else:
            self.model = self._initialize_model().to(self.device)

    def _initialize_model(self):
        if self.model_name == "autoencoder":
            return Autoencoder(**self.model_params)
        elif self.model_name == "lstm":
            return LSTMModel(**self.model_params)
        elif self.model_name == "cnn":
            return CNNModel(**self.model_params)
        elif self.model_name == "stackvaeg":
            return StackVAEG(**self.model_params)
        elif self.model_name == "omnianomaly":
            return OmniAnomaly(**self.model_params)
        elif self.model_name == "cnnautoencoder":
            return CnnAutoencoder(**self.model_params)
        elif self.model_name == "kan":
            return KAN(**self.model_params)
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

    def train(
        self,
        train_sequences=None,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        train_raw=None,
        validation_sequences=None,
        early_stopping_patience=None,
        reduce_lr_patience=None,
        reduce_lr_factor=0.1,
        min_lr=1e-6,
    ):
        if self.model_name == "egads":
            if train_raw is None:
                raise ValueError("EGADS requires 'train_raw'")
            data_np = train_raw.cpu().numpy().flatten()
            self.model.fit(data_np)
            return
        elif self.model_name == "dtw":
            if train_sequences is None:
                raise ValueError("DTW requires 'train_sequences'")
            self.model.fit(train_sequences)
            return

        if train_sequences is None:
            raise ValueError(f"{self.model_name} requires 'train_sequences'")

        train_loader = torch.utils.data.DataLoader(
            train_sequences, batch_size=batch_size, shuffle=True
        )
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        scheduler = None
        if reduce_lr_patience is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=reduce_lr_factor,
                patience=reduce_lr_patience,
                min_lr=min_lr,
            )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False
        best_model_weights = None

        self.model.train()
        running_loss = 0.0
        for epoch in tqdm(range(num_epochs), desc=f"Training {self.model_name}"):
            running_loss = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                if self.model_name == "cnnautoencoder":
                    if batch.ndimension() == 2:
                        batch = batch.unsqueeze(1)
                    elif batch.ndimension() == 3:
                        if batch.size(2) == 1:
                            batch = batch.permute(0, 2, 1)
                        else:
                            if batch.size(1) != 1:
                                raise ValueError(
                                    f"Expected channels=1, got {batch.size(1)}"
                                )
                    else:
                        raise ValueError(
                            f"Unexpected batch shape for 'cnnautoencoder': {batch.shape}"
                        )

                if self.model_name in ["stackvaeg", "omnianomaly"]:
                    if self.model_name == "stackvaeg":
                        outputs, mu, logvar = self.model(batch)
                        batch_flat = batch.view(batch.size(0), -1)
                        recon_loss = nn.MSELoss()(outputs, batch_flat)
                    else:
                        reconstructed, mu, logvar = self.model(batch)
                        recon_loss = nn.MSELoss()(reconstructed, batch)
                    # KLD
                    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kld * 0.001
                else:
                    outputs = self.model(batch)
                    if self.model_name == "cnn":
                        loss = nn.MSELoss()(outputs, batch.view_as(outputs))
                    elif self.model_name == "kan":
                        target = batch[:, -1, 0]
                        loss = nn.MSELoss()(outputs, target)
                    elif self.model_name == "lstm":
                        loss = nn.MSELoss()(outputs, batch)
                    elif self.model_name == "cnnautoencoder":
                        loss = nn.MSELoss()(outputs, batch)
                    else:
                        loss = nn.MSELoss()(outputs, batch.view(outputs.size()))

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

            val_loss = 0.0
            if validation_sequences is not None:
                val_loader = torch.utils.data.DataLoader(
                    validation_sequences, batch_size=batch_size, shuffle=False
                )
                self.model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        if self.model_name in ["stackvaeg", "omnianomaly"]:
                            if self.model_name == "stackvaeg":
                                outputs, mu, logvar = self.model(batch)
                                batch_flat = batch.view(batch.size(0), -1)
                                recon_loss = nn.MSELoss()(outputs, batch_flat)
                            else:
                                reconstructed, mu, logvar = self.model(batch)
                                recon_loss = nn.MSELoss()(reconstructed, batch)
                            # KLD
                            kld = -0.5 * torch.mean(
                                1 + logvar - mu.pow(2) - logvar.exp()
                            )
                            loss = recon_loss + kld * 0.001
                        else:
                            outputs = self.model(batch)
                            if self.model_name == "cnn":
                                loss = nn.MSELoss()(outputs, batch.view_as(outputs))
                            elif self.model_name == "kan":
                                target = batch[:, -1, 0]
                                loss = nn.MSELoss()(outputs, target)
                            elif self.model_name == "lstm":
                                loss = nn.MSELoss()(outputs, batch)
                            elif self.model_name == "cnnautoencoder":
                                loss = nn.MSELoss()(outputs, batch)
                            else:
                                loss = nn.MSELoss()(outputs, batch.view(outputs.size()))
                        val_loss += loss.item()
                val_loss /= len(val_loader)
                print(f"Validation Loss: {val_loss:.6f}")

            if scheduler:
                monitored_loss = (
                    val_loss if validation_sequences is not None else epoch_loss
                )
                scheduler.step(monitored_loss)

            if early_stopping_patience is not None:
                current_loss = (
                    val_loss if validation_sequences is not None else epoch_loss
                )
                if current_loss < best_val_loss:
                    best_val_loss = current_loss
                    epochs_no_improve = 0
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch+1}")
                        early_stop = True
            if early_stop:
                break

        if early_stopping_patience is not None and best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            print("Restored best model weights.")

    def predict(self, test_sequences=None, test_raw=None):
        if self.model_name == "egads":
            if test_raw is None:
                raise ValueError("EGADS requires 'test_raw'")
            scores = self.model.predict(test_raw.cpu().numpy().flatten())
            preds = np.zeros_like(scores)
            return scores, preds
        elif self.model_name == "dtw":
            if test_sequences is None:
                raise ValueError("DTW requires 'test_sequences'")
            scores = self.model.predict(test_sequences)
            preds = np.zeros_like(scores)
            return scores, preds

        self.model.eval()
        all_errors = []
        all_preds = []
        with torch.no_grad():
            for i in range(len(test_sequences)):
                x = test_sequences[i].unsqueeze(0).to(self.device)

                if self.model_name == "cnnautoencoder":
                    # Check current dimensions
                    if x.ndimension() == 2:
                        # Shape: [1, sequence_length]
                        x = x.unsqueeze(1)  # [1, 1, sequence_length]
                    elif x.ndimension() == 3:
                        # Shape: [batch_size, channels, sequence_length] or [batch_size, sequence_length, channels]
                        # Assuming [1, sequence_length, channels], permute to [1, channels, sequence_length]
                        if x.size(2) == 1:
                            x = x.squeeze(2)  # [1, sequence_length]
                            x = x.unsqueeze(1)  # [1, 1, sequence_length]
                        else:
                            raise ValueError(
                                f"Unexpected input shape for 'cnnautoencoder': {x.shape}"
                            )
                    else:
                        raise ValueError(
                            f"Unexpected input shape for 'cnnautoencoder': {x.shape}"
                        )
                if self.model_name in ["stackvaeg", "omnianomaly"]:
                    if self.model_name == "stackvaeg":
                        outputs, _, _ = self.model(x)
                        target = x.view(1, -1)
                        mse = torch.mean((outputs - target) ** 2, dim=1)
                        preds = outputs[:, -1:]
                        pred_val = preds.mean().item()
                    else:
                        reconstructed, _, _ = self.model(x)
                        mse = torch.mean((reconstructed - x) ** 2, dim=[1, 2])
                        pred_val = reconstructed[0, -1, 0].item()
                else:
                    outputs = self.model(x)
                    if self.model_name == "cnn":
                        mse = torch.mean(
                            (outputs - x.view_as(outputs)) ** 2, dim=[1, 2]
                        )
                        pred_val = outputs[0, -1].item()
                    elif self.model_name == "kan":
                        target = x[:, -1, 0]
                        mse = torch.mean((outputs - target) ** 2, dim=0, keepdim=False)
                        pred_val = outputs.item()

                    elif self.model_name == "lstm":
                        mse = torch.mean((outputs - x) ** 2, dim=[1, 2])
                        pred_val = outputs[0, -1, 0].item()
                    elif self.model_name == "cnnautoencoder":
                        mse = torch.mean((outputs - x) ** 2, dim=[1, 2])
                        pred_val = outputs[0, -1].item()
                    else:
                        # autoencoder
                        mse = torch.mean((outputs - x.view(1, -1)) ** 2, dim=1)
                        if outputs.dim() == 2:
                            pred_val = outputs[0, -1].item()
                        else:
                            pred_val = outputs.mean().item()
                all_errors.append(mse.item())
                all_preds.append(pred_val)
        return np.array(all_errors), np.array(all_preds)

    def get_anomalies(self, errors, percentile=99.5):
        threshold = np.percentile(errors, percentile)
        anomalies = errors > threshold
        return anomalies, threshold
