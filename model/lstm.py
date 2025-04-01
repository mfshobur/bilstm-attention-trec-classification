import torch
import torch.nn as nn
from typing import Tuple, Optional

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_x = nn.Linear(input_size, 4 * hidden_size) # 4 is for forget, input, gate, and output gate
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size)
    
    def forward(self, x, hx):
        assert x.shape[-1] == self.W_x.in_features, "Input size mismatch"
        assert hx[0].shape[-1] == self.W_x.out_features // 4, "Output size mismatch"

        gates = self.W_x(x) + self.W_h(hx[0])
        f, i, g, o = torch.chunk(gates, 4, dim=-1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * hx[1] + i * g

        h = o * torch.tanh(c)

        return h, c

class LSTM(nn.Module):
    """
    # Custom LSTM implementation built from scratch using LSTMCell.
    
    This implementation supports multiple layers and bidirectional processing.
    It processes sequences by applying an LSTM cell at each time step and
    propagating the hidden states across layers.
    """
    
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1, 
                 bidirectional: bool = False, 
                 device: Optional[torch.device] = None):
        """
        # Initialize the LSTM module with configurable parameters.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of stacked LSTM layers
            bidirectional: Whether to use bidirectional processing
            device: Device to place tensors on (CPU/GPU)
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.device = device
        self.layers_backward = None

        # Create forward LSTM layers with appropriate input sizes
        self.layers_forward = nn.ModuleList(
            [LSTMCell(input_size, hidden_size) if i == 0 
             else LSTMCell(hidden_size, hidden_size) 
             for i in range(num_layers)]
        )
        
        # Create backward LSTM layers if bidirectional
        if bidirectional:
            self.layers_backward = nn.ModuleList(
                [LSTMCell(input_size, hidden_size) if i == 0 
                 else LSTMCell(hidden_size, hidden_size) 
                 for i in range(num_layers)]
            )
    
    def forward(self, 
                x: torch.Tensor, 
                h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        # Forward pass of the LSTM module.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            h0: Optional initial hidden state tuple (h, c) of shape 
                (num_layers * num_directions, batch_size, hidden_size)
                
        Returns:
            Tuple containing:
            - output: Tensor of shape (batch_size, seq_length, hidden_size * num_directions)
            - h_n: Final hidden state of shape (num_layers * num_directions, batch_size, hidden_size)
            - c_n: Final cell state of shape (num_layers * num_directions, batch_size, hidden_size)
        
        # Note: This implementation manually handles the sequence processing that
        # PyTorch's built-in LSTM would do automatically.
        """
        # Validate input dimensions
        assert x.shape[-1] == self.input_size, f"Input's last dimension did not match. Expected ({self.input_size}) but got ({x.shape[-1]}) instead"

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Initialize hidden and cell states
        if h0 is None:
            # Create zero tensors for initial states if not provided
            hx = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)
            cx = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)
        else:
            # Validate provided initial states
            assert h0[0].shape == (self.num_layers * self.num_directions, batch_size, self.hidden_size), \
                f"h0 shape did not match. Expected ({self.num_layers * self.num_directions, batch_size, self.hidden_size}) but got ({h0[0].shape}) instead"
            hx, cx = h0[0], h0[1]

        # Pre-allocate output tensors
        output = torch.zeros(batch_size, seq_length, self.hidden_size * self.num_directions, device=self.device)
        h_n = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)
        c_n = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=self.device)

        # Process each layer
        for layer in range(self.num_layers):
            # Initialize layer-specific variables
            if layer == 0:
                # For first layer, input is the original sequence
                input_forward = x
                layer_outputs_forward = torch.zeros(batch_size, seq_length, self.hidden_size, device=self.device)
                
                if self.bidirectional:
                    # For bidirectional, flip the sequence on time dimension
                    input_backward = torch.flip(x, dims=(1,))
                    layer_outputs_backward = torch.zeros(batch_size, seq_length, self.hidden_size, device=self.device)

            # Get initial hidden states for this layer
            ht_forward = hx[layer]
            ct_forward = cx[layer]
            
            if self.bidirectional:
                # Index calculation for bidirectional hidden states
                backward_idx = self.num_layers + layer
                ht_backward = hx[backward_idx]
                ct_backward = cx[backward_idx]
            
            # Process each time step in the sequence
            for seq in range(seq_length):
                # Forward pass for this time step
                ht_forward, ct_forward = self.layers_forward[layer](input_forward[:,seq], (ht_forward, ct_forward))
                layer_outputs_forward[:,seq] = ht_forward

                if self.bidirectional:
                    # Backward pass for this time step
                    ht_backward, ct_backward = self.layers_backward[layer](input_backward[:,seq], (ht_backward, ct_backward))
                    layer_outputs_backward[:,seq] = ht_backward

            # Prepare inputs for next layer
            # .clone() creates a copy
            input_forward = layer_outputs_forward.clone()
            
            if self.bidirectional:
                input_backward = layer_outputs_backward.clone()
                
            # Store final hidden states for this layer
            h_n[layer] = ht_forward
            c_n[layer] = ct_forward
            
            if self.bidirectional:
                h_n[backward_idx] = ht_backward
                c_n[backward_idx] = ct_backward
                
        # Combine outputs from forward and backward passes
        output[:, :, :self.hidden_size] = input_forward
        
        if self.bidirectional:
            # Flip backward outputs to align with forward sequence
            input_backward_aligned = torch.flip(input_backward, dims=(1,))
            output[:, :, self.hidden_size:] = input_backward_aligned
        
        # Validate output shapes
        assert output.shape == (batch_size, seq_length, self.hidden_size * self.num_directions), \
            f"Output shape mismatch. Expected ({batch_size, seq_length, self.hidden_size * self.num_directions}) but got ({output.shape}) instead"
        assert h_n.shape == (self.num_layers * self.num_directions, batch_size, self.hidden_size), \
            f"h_n shape mismatch. Expected ({self.num_layers * self.num_directions, batch_size, self.hidden_size}) but got ({h_n.shape}) instead"
            
        return output, (h_n, c_n)
    
if __name__ == "__main__":
    pass