import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
#         self.nn1 = nn.Linear(in_features = state_size, out_features = state_size*32)
#         self.nn2 = nn.Linear(in_features = state_size*32, out_features = state_size*8)
#         self.nn3 = nn.Linear(in_features = state_size*8, out_features = action_size)
        self.nn1 = nn.Linear(in_features = state_size, out_features = 512)
        self.nn2 = nn.Linear(in_features = 512, out_features = 128)
        self.nn3 = nn.Linear(in_features = 128, out_features = action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.leaky_relu(self.nn1(state))
        x = F.leaky_relu(self.nn2(x))
        x = self.nn3(x)
        
        return x
        
