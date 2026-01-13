
import torch
import torch.nn as nn
from ultralytics.nn.modules.block import PSABlock

 
__all__ = ['GL_CAB', 'GL_CAB_PSABlock']
 
class GL_CAB(nn.Module):

    """
    Implementation of the Global-Local Combined Attention Block (GL-CAB)
    from the paper "Global-Local Attention Mechanism Based Small Object Detection".
    
    This module creates an attention map by combining global context, local features,
    and local detail features to emphasize small objects that might be lost.
    """
    def __init__(self, c1, c2=None, n=2):
        # c1: input channels
        # c2: output channels (defaults to c1 if not provided)
        super(GL_CAB, self).__init__()
        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2
        self.n = n
            
        # A reusable 1x1 convolution block with GroupNorm (more stable than BatchNorm for small batches)
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )

        # --- Path for Global Features: G(P) in the paper ---
        # Captures the overall scene context.
        self.global_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.ReLU()
        )

        # --- Path for Local Features: Z(P) in the paper ---
        # Focuses on local information without global context.
        self.local_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.SiLU()
        )

        # --- Path for Local Detail Features: L(P) in the paper ---
        # Refines details using global context as a guide.
        # This path takes the output of the global average pooling.
        self.local_detail_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            conv_block(c1, c1),
            nn.SiLU()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Final convolution to ensure the output channel count is correct
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x,n=2):
        """
        Forward pass of the GL-CAB module.
        Follows Formula (5): W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)
        The attention map is applied to the globally-processed features.
        """
        # P is the input tensor x
        
        # G(P): Processed global features
        global_features_processed = self.global_features_path(x)
        
        # L(P): Local details derived from global context
        local_detail_features = self.local_detail_path(x)
        
        # Z(P): Processed local features
        local_features_processed = self.local_features_path(x)
        
        # L(P) ⊕ Z(P): Fusion of local and detail features (element-wise addition)
        fused_local_info = local_detail_features + local_features_processed
        
        # σ(...): The sigmoid function creates the final attention map
        attention_map = self.sigmoid(fused_local_info)
        
        # ⊗ G(P): The attention map is applied to the processed global features
        out = attention_map * global_features_processed
        
        return self.final_conv(out)
    

class GL_CAB_PSABlock(nn.Module):

    """
    Implementation of the Global-Local Combined Attention Block (GL-CAB)
    from the paper "Global-Local Attention Mechanism Based Small Object Detection".
    
    This module creates an attention map by combining global context, local features,
    and local detail features to emphasize small objects that might be lost.
    """
    def __init__(self, c1, c2=None, n=2):
        # c1: input channels
        # c2: output channels (defaults to c1 if not provided)
        super(GL_CAB_PSABlock, self).__init__()
        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2
        self.n = n

        self.m = nn.Sequential(*(PSABlock(self.c1, attn_ratio=0.5, num_heads=self.c1 // 64) for _ in range(n)))
            
        # A reusable 1x1 convolution block with GroupNorm (more stable than BatchNorm for small batches)
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels)
            )

        # --- Path for Global Features: G(P) in the paper ---
        # Captures the overall scene context.
        self.global_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.ReLU()
        )

        # --- Path for Local Features: Z(P) in the paper ---
        # Focuses on local information without global context.
        self.local_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.SiLU()
        )

        # --- Path for Local Detail Features: L(P) in the paper ---
        # Refines details using global context as a guide.
        # This path takes the output of the global average pooling.
        self.local_detail_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            conv_block(c1, c1),
            nn.SiLU()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Final convolution to ensure the output channel count is correct
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x,n=2):
        """
        Forward pass of the GL-CAB module.
        Follows Formula (5): W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)
        The attention map is applied to the globally-processed features.
        """
        # P is the input tensor x
        
        # G(P): Processed global features
        global_features_processed = self.global_features_path(x)

        global_features_processed = self.m(global_features_processed)
        
        # L(P): Local details derived from global context
        local_detail_features = self.local_detail_path(x)
        
        # Z(P): Processed local features
        local_features_processed = self.local_features_path(x)
        
        # L(P) ⊕ Z(P): Fusion of local and detail features (element-wise addition)
        fused_local_info = local_detail_features + local_features_processed
        
        # σ(...): The sigmoid function creates the final attention map
        attention_map = self.sigmoid(fused_local_info)
        
        # ⊗ G(P): The attention map is applied to the processed global features
        out = attention_map * global_features_processed
        
        return self.final_conv(out)
 
 
if __name__ == "__main__":
    # Generating Sample image
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)
 
    # Model
    gl_cab = GL_CAB(64, 64)
    out = gl_cab(image)
    print(out.size())

    gl_cab_psa_block = GL_CAB_PSABlock(64, 64)

 
    out = gl_cab_psa_block(image)
    print(out.size())