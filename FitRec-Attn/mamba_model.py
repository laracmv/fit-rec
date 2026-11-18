"""Mamba variant of FitRec (`python fitrect-attn.py --arch mamba`).

The two attention LSTMs of FitRec-Attn are replaced by a stack of Mamba blocks
(Gu & Dao, 2023, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces").
Everything else stays the same: the data windows (T = 10), the user embedding, the
previous-workout context encoder, the MSE loss and the training loop.

Each of the T steps of a window becomes one token:

    token_t = [speed_t, altitude_t, user embedding, context_t, HR_{t-1}, HR_known]

so the token of the last step carries the inputs of the step being predicted and the
last observed heart rate. The Mamba stack reads the tokens left to right (causal) and a
linear head on the last token predicts the heart rate of step T.

The block is plain PyTorch, so it runs on any GPU (and on CPU) without compiling
anything. If the `mamba-ssm` package is installed, its fused CUDA kernel computes the
same selective scan faster and with less memory; the parameters are identical, so a
checkpoint works with both.
"""

import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

try:  # optional fast path
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # not installed, or no CUDA build
    selective_scan_fn = None


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaMixer(nn.Module):
    """The selective SSM layer of a Mamba block (Gu & Dao 2023, Algorithm 2 + Figure 3)."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-3, dt_max=0.1):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)  # x and the gate z
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )  # causal depthwise convolution (the extra right padding is cut off)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)  # dt, B, C
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # initialization of the reference implementation: dt in [dt_min, dt_max] after softplus,
        # A = -[1..d_state] for every channel (S4D-real), D = 1
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse of softplus
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x):
        # x: batch * L * d_model
        L = x.size(1)
        x, z = self.in_proj(x).chunk(2, dim=-1)  # batch * L * d_inner each
        x = self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)

        dt, B, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.transpose(1, 2)  # batch * d_inner * L, bias added in the scan
        A = -torch.exp(self.A_log.float())  # d_inner * d_state, negative so the state decays

        if selective_scan_fn is not None and x.is_cuda:
            y = selective_scan_fn(
                x.transpose(1, 2).contiguous(), dt.contiguous(), A,
                B.transpose(1, 2).contiguous(), C.transpose(1, 2).contiguous(), self.D.float(),
                z=None, delta_bias=self.dt_proj.bias.float(), delta_softplus=True,
            ).transpose(1, 2)
        else:
            y = self.selective_scan(x, F.softplus(dt.transpose(1, 2) + self.dt_proj.bias), A, B, C)

        return self.out_proj(y * F.silu(z))

    def selective_scan(self, x, dt, A, B, C):
        """h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t ;  y_t = C_t h_t + D x_t  (sequential over L)."""
        h = x.new_zeros(x.size(0), self.d_inner, self.d_state)
        ys = []
        for t in range(x.size(1)):
            dt_t = dt[:, t, :, None]  # batch * d_inner * 1
            h = torch.exp(dt_t * A) * h + (dt_t * x[:, t, :, None]) * B[:, t, None, :]
            ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(ys, dim=1) + x * self.D


class MambaBlock(nn.Module):
    """Residual block: x + Mixer(RMSNorm(x))."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state=d_state)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaEncoder(nn.Module):
    """Builds one token per step and runs the Mamba stack (replaces encoder + decoder LSTMs)."""

    def __init__(self, input_size, d_model, T, attr_embeddings, hr_mean, hr_std, n_layers=2, d_state=16, checkpoint=False):
        super().__init__()
        self.T = T
        # recompute each block in the backward pass instead of keeping its activations:
        # ~half the GPU memory, ~30% slower, same result (needed for T = 20 on a 16 GB GPU)
        self.checkpoint = checkpoint
        self.user_embedding = attr_embeddings[0]  # same names as the LSTM encoder, so the
        self.sport_embedding = attr_embeddings[1]  # optimizer's weight-decay groups still match
        self.in_proj = nn.Linear(input_size + 2, d_model)  # + previous HR + "previous HR is known"
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d_model)
        # heart rate is in bpm (not scaled by the data reader); standardize it inside the model
        self.register_buffer("hr_mean", torch.tensor(float(hr_mean)))
        self.register_buffer("hr_std", torch.tensor(float(hr_std)))

    def forward(self, attr_inputs, context_embedding, input_variable, y_history):
        # same feature concatenation as the LSTM encoder
        for attr, attr_input in attr_inputs.items():
            emb = self.user_embedding if attr == "user_input" else self.sport_embedding
            input_variable = torch.cat([emb(attr_input), input_variable], dim=-1)
        if context_embedding is not None:
            input_variable = torch.cat([context_embedding, input_variable], dim=-1)

        # token t sees the heart rate of step t-1; step 0 has none (0 + flag 0)
        y = (y_history - self.hr_mean) / self.hr_std  # batch * (T-1)
        prev_hr = F.pad(y, (1, 0))  # batch * T
        known = torch.ones_like(prev_hr)
        known[:, 0] = 0
        tokens = torch.cat([input_variable, prev_hr.unsqueeze(-1), known.unsqueeze(-1)], dim=-1)

        h = self.in_proj(tokens)
        for layer in self.layers:
            if self.checkpoint and self.training and torch.is_grad_enabled():
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        return self.norm_f(h)  # batch * T * d_model


class MambaHead(nn.Module):
    """Linear read-out of the last token, back in bpm."""

    def __init__(self, d_model, hr_mean, hr_std):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        self.register_buffer("hr_mean", torch.tensor(float(hr_mean)))
        self.register_buffer("hr_std", torch.tensor(float(hr_std)))

    def forward(self, h):
        return self.fc(h[:, -1]).squeeze(-1) * self.hr_std + self.hr_mean
