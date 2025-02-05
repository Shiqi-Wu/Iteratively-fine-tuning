import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .adapter import Adapter


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, x):
        # This code implements the forward propagation process of the multi-head self-attention mechanism.
        
        # Obtain the shape of the input tensor x, where B is the batch size, N is the sequence length, and C is the feature dimension.
        B, N, C = x.shape 

        #Apply linear transformation to the input tensor x through the query (Query) projection layer
        q = self.q_proj(x)

        # Apply linear transformation to the input tensor x through the key (Key) projection layer self.k_proj, and then use the self._shape function to adjust the tensor shape to (B * num_heads, N, head_dim), where num_heads is the number of heads in the multi-head self-attention mechanism, and head_dim is the feature dimension of each head.
        k = self._shape(self.k_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        # Calculate the attention weights: 
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        # Perform softmax normalization on the attention weights to ensure that the sum of the weights is 1.
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)
        
        # Adjust the shape of the attention output tensor to (B, num_heads, N, head_dim)
        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        # x = W*a
        x = self.proj(attn_output)
        # drop out
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            if self.config.adpnum_option == 'multi':
                self.adaptmlp = nn.ModuleList()
                for i, ffn_num in enumerate(config.ffn_num):
                    adapter = Adapter(self.config, dropout=0.1, bottleneck=ffn_num,
                                  init_option=config.ffn_adapter_init_option,
                                  adapter_scalar=config.ffn_adapter_scalar,
                                  adapter_layernorm_option=config.ffn_adapter_layernorm_option)
                    self.adaptmlp.append(adapter)

            elif self.config.adpnum_option == 'single':
                self.adaptmlp = Adapter(self.config, dropout=0.1, bottleneck=config.ffn_num[0],
                                    init_option=config.ffn_adapter_init_option,
                                    adapter_scalar=config.ffn_adapter_scalar,
                                    adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                                    )        
            else:
                raise ValueError(self.config.ffn_adapt_num)    

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.config.ffn_adapt and self.config.ffn_option == 'parallel':
            if self.config.adpnum_option == 'multi':
                adapt_x = []
                for adaptmlp in self.adaptmlp:
                    adapt_x.append(adaptmlp(x, add_residual = False))                
            elif self.config.adpnum_option == 'single':
                adapt_x = self.adaptmlp(x, add_residual=False)
            else:
                raise ValueError(self.config.ffn_adapt_num)


        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            # if self.config.ffn_option == 'sequential':
                # x = self.adaptmlp(x)
            # Only consider the paralled type
            if self.config.ffn_option == 'parallel':
                if self.config.adpnum_option == 'multi':
                    for adapt_value in adapt_x:
                        x = x + adapt_value
                elif self.config.adpnum_option == 'single':
                    x = x + adapt_x
                else:
                    raise ValueError(self.config.ffn_adapt_num)
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        return x
