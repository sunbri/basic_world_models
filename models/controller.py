import torch
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    """
    Simple linear model
    [z (32) + h (512)] -> Action (3)
    """
    def __init__(self, z_size=32, h_size=512, a_size=3):
        super().__init__()
        self.input_size = z_size + h_size
        self.fc = nn.Linear(self.input_size, a_size)

    def forward(self, z, h):
        input = torch.cat([z, h], dim=1)
        output = self.fc(input)
        # clip the values into the correct range
        steer = torch.tanh(output[:, 0])
        gas = torch.clamp(torch.tanh(output[:, 1]), min=0.0)
        brake = torch.clamp(torch.tanh(output[:, 2]), min=0.0)
        return torch.stack([steer, gas, brake], dim=1)

    def get_parameters_flat(self):
        return np.concatenate([p.data.cpu().numpy().flatten() for p in self.parameters()])

    def set_parameters_flat(self, params):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            new_p = params[offset : offset + numel]
            p.data = torch.from_numpy(new_p).float().to(p.device).view(p.size())
            offset += numel