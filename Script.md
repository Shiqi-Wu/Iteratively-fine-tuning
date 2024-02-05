## Vision Transformer Structure

### Patch Embedding
- **PatchEmbed**
  - `proj`: Conv2d with 3 input channels, 768 output channels, kernel size of (16, 16), and stride of (16, 16)
  - `norm`: Identity (no normalization)

### Positional Dropout
- **pos_drop**: Dropout with a probability (p) of 0.0

### Blocks(12)
- **Sequential**
  - Each `Block` consists of the following components:
    - `norm1`: LayerNorm with 768 features and an epsilon of 1e-06
    - `attn`: Attention module with:
      - `q_proj`, `v_proj`, `k_proj`: Linear layers with 768 input and output features
      - `attn_drop`: Dropout with p=0.0
      - `proj`: Linear layer with 768 input and output features
      - `proj_drop`: Dropout with p=0.0
    - `drop_path`: Identity (no path dropout)
    - `norm2`: LayerNorm with 768 features and an epsilon of 1e-06
    - `fc1`: Linear layer with 768 input features and 3072 output features
    - `fc2`: Linear layer with 3072 input features and 768 output features
    - `act`: GELU activation function
    - `mlp_drop`: Dropout with p=0.0
    - `adaptmlp`: ModuleList containing 2 x Adapter modules, each with:
      - `down_proj`: Linear layer with 768 input features and 32 output features
      - `non_linear_func`: ReLU activation function
      - `up_proj`: Linear layer with 32 input features and 768 output features

### Final Layers
- **norm**: LayerNorm with 768 features and an epsilon of 1e-06
- **pre_logits**: Identity (no transformation before logits)
- **head**: Sequential containing:
  - BatchNorm1d with 768 features, an epsilon of 1e-06, momentum of 0.1, affine set to False, and tracking running stats
  - Linear layer with 768 input features and 100 output features

This model architecture follows the typical structure of a Vision Transformer, with a patch embedding layer to convert input images into a sequence of flattened patches, a series of transformer blocks for feature extraction, and a head for classification. The use of adapters (`adaptmlp`) indicates a modular approach to potentially enhance the model's ability to learn task-specific features or facilitate transfer learning. The attention mechanism within each block allows the model to weigh the importance of different patches when forming representations, which is a key advantage of transformer architectures in vision tasks.