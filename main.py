import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Config:
    D: int = 768 # Model Width
    D_RNN: int = 1024 # RNN Width
    N: int = 12 # Depth
    M: int = 3 # MLP expansion factor
    mult: int = 4 # Multiplier of D_RNN
    vocab_size: int = 1


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return nn.functional.normalize(x, dim=-1) * self.scale * self.g


class GatedMLPBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.D = config.D
        self.M = config.M
        self.linear_left = nn.Linear(self.D, self.D * self.M)
        self.linear_right = nn.Linear(self.D, self.D * self.M)
        self.out_linear = nn.Linear(self.D * self.M, self.D)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_left = self.linear_left(x)
        x_right = self.linear_right(x)
        return self.out_linear(self.gelu(x_left) * x_right)


class RGLRU(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.hidden_dim = config.D_RNN # * config.mult
        self.Wa = nn.Parameter(data= torch.Tensor(self.hidden_dim, config.D_RNN))
        self.Wx = nn.Parameter(data=torch.Tensor(self.hidden_dim, config.D_RNN))
        self.ba = nn.Parameter(data=torch.Tensor(self.hidden_dim))
        self.bx = nn.Parameter(data=torch.Tensor(self.hidden_dim))
        self.c = 8 # The paper suggested this value page: 4
        self.lmbd = nn.Parameter(data=torch.Tensor(self.hidden_dim))

        self.reset_parameters()

        self.a = nn.functional.sigmoid(self.lmbd) # page: 4

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.Wa, nonlinearity="linear", mode="fan_in")
        nn.init.kaiming_normal_(self.Wx, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.ba, val=0)
        nn.init.constant_(self.bx, val=0)
        nn.init.uniform_(self.lmbd, a=0.9, b=0.999)


    def forward(self, x):
        _, L, _ = x.shape
        h = torch.zeros_like(self.ba)
        y = []
        for t in range(L):
            xt = x[:, t, :]
            rt = nn.functional.sigmoid(nn.functional.linear(xt, self.Wa, self.ba)) # eqn(1)
            it = nn.functional.sigmoid(nn.functional.linear(xt, self.Wx, self.bx)) # eqn(2)
            at = torch.pow(self.a, self.c * rt) # eqn(3)
            h = at * h + torch.sqrt(1 - at.pow(2)) * (it * xt) # eqn(4)
            y.append(h.unsqueeze(1))

        y = torch.cat(y, dim=1)
        return y


class RecurrentBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.left_linear = nn.Linear(config.D, config.D_RNN)
        self.right_linear = nn.Linear(config.D, config.D_RNN)
        self.gelu = nn.GELU()
        self.rglru = RGLRU(config)
        self.out_linear = nn.Linear(config.D_RNN, config.D) 

    def forward(self, x):

        _, S, _ = x.shape
        x_left = self.left_linear(x)
        x_left = self.gelu(x_left)
        x_right = self.right_linear(x)
        x_right = nn.Conv1d(in_channels=S, out_channels=S, kernel_size=3, padding=1)(x_right)
        x_right = self.rglru(x_right)
        return self.out_linear(x_left * x_right)

class ResidualBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.rmsnorm1 = RMSNorm(config.D)
        self.rmsnorm2 = RMSNorm(config.D)
        self.temporal_mixing_block = RecurrentBlock(config=config)
        self.mlp = GatedMLPBlock(config=config)
        
    def forward(self, x):
        res1 = x
        x = self.rmsnorm1(x)
        x = self.temporal_mixing_block(x)
        x = res1 + x

        res2 = x
        x = self.rmsnorm2(x)
        x = self.mlp(x)
        return res2 + x

class Hawk(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(config.vocab_size, config.D)
        self.blocks = nn.ModuleList([ResidualBlock(config=config) for i in range(config.N)])
        self.lm_head = nn.Linear(config.D, config.vocab_size)

    def forward(self, x):
        x = self.word_embedding(x)
        for block in self.blocks:
            x = block(x)

        return self.lm_head(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    config = Config()
    x = torch.randint(0, config.vocab_size, (1, 100))
    model = Hawk(config)
    y = model(x)
    print(y.shape)

    parameter_count = count_parameters(model)

    print(f"Parameter Numbers of Model = {parameter_count}")