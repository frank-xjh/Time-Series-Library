import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class CF_Inference_Net(nn.Module):
    """
    Change Factor Inference Network (CF_Inference_Net).
    This network infers the latent change factors (theta) from the input time series.
    It outputs the parameters (mu and log_var) of a Gaussian distribution for the latent factors.
    """
    def __init__(self, n_vars, seq_len, theta_dim, hidden_dim=128):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(n_vars * seq_len, hidden_dim)
        self.relu = nn.ReLU()
        # Output layers for mu and log_var
        self.fc_mu = nn.Linear(hidden_dim, theta_dim)
        self.fc_log_var = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x):
        # x: [bs, seq_len, n_vars]
        x = x.transpose(1, 2) # -> [bs, n_vars, seq_len]
        x = self.flatten(x)   # -> [bs, n_vars * seq_len]

        hidden = self.relu(self.fc1(x))

        mu = self.fc_mu(hidden)         # [bs, theta_dim]
        log_var = self.fc_log_var(hidden) # [bs, theta_dim]

        return mu, log_var

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        self.theta_dim = 32 # A new hyperparameter, e.g., 32
        # 1. Change Factor Inference Network
        self.cf_inference_net = CF_Inference_Net(configs.enc_in, configs.seq_len, self.theta_dim)
        # 2. Projection layer to make theta compatible with patch embeddings
        self.theta_projector = nn.Linear(self.theta_dim, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forecast(self, x_enc, theta): # MODIFIED: Accept theta as an argument
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs = x_enc.shape[0]
        x_enc = x_enc.permute(0, 2, 1) # -> [bs, n_vars, seq_len]

        # Patching and embedding
        enc_out, n_vars = self.patch_embedding(x_enc) # enc_out: [bs * nvars, patch_num, d_model]

        # ------ NEW: CONDITIONING THE ENCODER ON THETA ------
        # Project theta to match d_model
        projected_theta = self.theta_projector(theta) # [bs, d_model]

        # Reshape and repeat projected_theta to add to each patch embedding
        # Repeat for each variable (n_vars)
        projected_theta = projected_theta.unsqueeze(1).repeat(1, n_vars, 1) # [bs, n_vars, d_model]
        # Reshape to match enc_out's batch dimension
        projected_theta = projected_theta.view(bs * n_vars, 1, self.d_model) # [bs * nvars, 1, d_model]

        # Add theta representation to the patch embeddings
        enc_out = enc_out + projected_theta # Broadcasting adds it to each patch
        # ----------------------------------------------------

        # Encoder
        enc_out, attns = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])) # [bs, nvars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2) # [bs, nvars, d_model, patch_num]

        # Decoder
        dec_out = self.head(enc_out)  # [bs, nvars, target_window]
        dec_out = dec_out.permute(0, 2, 1) # [bs, target_window, nvars]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
             # ------ NEW: VAE FORWARD PASS LOGIC ------
            # 1. Infer latent factor distribution from input
            mu, log_var = self.cf_inference_net(x_enc)
            # 2. Sample theta from the latent distribution using reparameterization trick
            theta = self.reparameterize(mu, log_var)
            # 3. Get the forecast conditioned on the sampled theta
            dec_out = self.forecast(x_enc, theta)
            # 4. Return a dictionary with prediction and latent variables for loss calculation
            return {
                "prediction": dec_out[:, -self.pred_len:, :], # [B, L, D]
                "mu": mu,
                "log_var": log_var
            }
            # ---------------------------------------------
        return None
