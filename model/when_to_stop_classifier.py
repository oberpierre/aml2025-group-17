from torch import nn

model = nn.Sequential(
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)