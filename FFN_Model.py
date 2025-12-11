"""
Contains the main FFN network for this project: ResidualFFNBinaryClassifier

Pretty standard 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNBinaryClassifier(nn.Module):
    """
    Simple Feed-Forward Network for binary classification.
    Takes the last non-padded hidden state and passes it through FFN layers.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: list = [512, 256],
        dropout: float = 0.1,
        use_batch_norm: bool = False,
    ):
        """
        Args:
            in_dim: Input dimension (hidden state size)
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256])
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        
        # Build FFN layers
        layers = []
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        """
        Extract last non-padded hidden state and pass through FFN.
        """

        # Pass through FFN
        logits = self.network(x).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)
        
        out = {"logits": logits, "probs": probs}
        
        if labels is not None:
            # Ensure labels are [B] and float
            labels = labels.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            out["loss"] = loss
        
        return out


class ResidualFFNBinaryClassifier(nn.Module):
    """
    FFN with residual connections for deeper networks.
    Includes residual scaling and proper initialization for stability.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        """
        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension (same for all layers)
            num_layers: Number of residual blocks
            dropout: Dropout probability
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection if dimensions don't match, we force this to happen.
        self.input_proj = None
        if in_dim != hidden_dim:
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            # Initialize input projection properly
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        
        # Residual blocks with scaled initialization
        # Scale decreases with depth to maintain signal magnitude
        scale_init = 1.0 / (num_layers ** 0.5)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout, use_layer_norm, scale_init=scale_init)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, 1)
        # Initialize output layer
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        
        # Input projection if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        logits = self.output(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        
        out = {"logits": logits, "probs": probs}
        
        #We calculate loss within forward pass, the model is designated for this task so this can be expected behavior.
        #We don't force a 'loss' key in output dictionary but it's my code so if I call index 'loss' in other code I'm an idiot
        if labels is not None:
            labels = labels.view(-1).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            out["loss"] = loss
        
        return out


class ResidualBlock(nn.Module):
    """Single residual block with LayerNorm and scaled residual connection.
        Uses duel drop outs pre and post down scaling"""
    def __init__(self, dim: int, dropout: float, use_layer_norm: bool, scale_init: float = 1.0):
        super().__init__()
        #Returns orginal tensor if no normilization is included, norm defaults to true and probably shouldn't be changed
        self.norm = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),  
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        # Learnable scale for residual connection
        self.scale = nn.Parameter(torch.ones(1) * scale_init)
        
        # Initialize the FFN layers properly
        # First linear: standard initialization
        nn.init.xavier_uniform_(self.ffn[0].weight)
        nn.init.zeros_(self.ffn[0].bias)
        
        # Second linear (output): initialize to small values for stability
        # This makes the residual branch contribute less initially
        nn.init.xavier_uniform_(self.ffn[3].weight, gain=0.1)
        nn.init.zeros_(self.ffn[3].bias)
    
    def forward(self, x):
        # Scaled residual connection: x + scale * f(norm(x))
        return x + self.scale * self.ffn(self.norm(x))