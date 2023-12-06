# --------------------------------------------------------
# References:
# https://github.com/jxhe/unify-parameter-efficient-tuning
# --------------------------------------------------------

import math
import torch
import torch.nn as nn

# Define the Adapter class, inheriting from nn.Module, which is a base class for building neural network modules in PyTorch.
class Adapter(nn.Module):
    # The initialization method for the Adapter class with several parameters for configuration.
    def __init__(self,
                 config=None,                 # 'config' is an optional parameter for model configuration.
                 d_model=None,                # 'd_model' represents the dimension of the model, can be specified or obtained from 'config'.
                 bottleneck=None,             # 'bottleneck' defines the bottleneck dimension for the adapter.
                 dropout=0.0,                 # 'dropout' specifies the dropout rate to be used in the adapter.
                 init_option="bert",          # 'init_option' is the method for initializing the adapter's parameters.
                 adapter_scalar="1.0",        # 'adapter_scalar' is a scaling factor for the adapter's output.
                 adapter_layernorm_option="in"):  # 'adapter_layernorm_option' determines the position of LayerNorm in the adapter.
        super().__init__()  # Initialize the superclass (nn.Module) constructor.
        # Set the number of embedding dimensions, either from the 'config' or using the provided 'd_model'.
        self.n_embd = config.d_model if d_model is None else d_model
        # Set the bottleneck size, either from the 'config' or using the provided 'bottleneck'.
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # Before adapter configuration.
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        # Initialize a LayerNorm layer before the adapter if specified in the configuration.
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        # Initialize 'scale' as a learnable parameter if specified, otherwise, use the provided scalar value.
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        # Linear layer to project the input down to the bottleneck size.
        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        # Non-linear function, ReLU in this case, for the adapter.
        self.non_linear_func = nn.ReLU()
        # Linear layer to project the input back up from the bottleneck size.
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        # Dropout rate for the adapter.
        self.dropout = dropout
        # Initialize the adapter weights based on the specified 'init_option'.
        if init_option == "bert":
            raise NotImplementedError  # 'bert' initialization option is not implemented.
        elif init_option == "lora":
            # Initialize weights using Kaiming uniform initialization and biases to zero for the 'lora' option.
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    # Forward method for the Adapter class, defining how input data 'x' passes through the adapter.
    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual  # Set 'residual' to input 'x' if it's not provided.
        # Apply LayerNorm before adapter processing if specified.
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        # Project input 'x' down to the bottleneck size.
        down = self.down_proj(x)
        # down = self.non_linear_func(down)  # Apply non-linear function (ReLU).
        # Apply dropout to the down-projected input.
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        # Project thçe input back up from the bottleneck size.
        up = self.up_proj(down)

        up = up * self.scale  # Scale the up-projected input by the 'scale' factor.

        # Apply LayerNorm after adapter processing if specified.
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        # Add the adapter output to the residual if 'add_residual' is True.
        if add_residual:
            output = up + residual
        else:
            output = up

        return output  # Return the output of the adapter module.
