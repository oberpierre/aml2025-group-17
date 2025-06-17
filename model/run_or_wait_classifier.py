import torch
import torch.nn as nn

class ConfidenceScorer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        input_dim = num_tags * 2  # get the mean and std together as a single input vector
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  
        )
    
    def forward(self, logits):
        # logits: (seq_len, num_tags)
        mean = logits.mean(dim=0)
        std = logits.std(dim=0)
        x = torch.cat([mean, std], dim=-1)
        return self.fc(x).squeeze() 