import torch

class confidence_model(torch.nn.Module):
    def __init__(self):
        super(confidence_model, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)