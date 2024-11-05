import torch
import einops
import torch.nn as nn
    
class QueCC(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper.
    From modeling_clip.py and modeling_vit.py in transformers repo
    """
    
    def __init__(self, tower_config, training_config):
        super().__init__()
        self.tower_config = tower_config
        self.training_config = training_config

        self.visual_token_count = getattr(training_config, 'mm_vision_output_token_count', 
                576 if getattr(training_config, 'mm_vision_select_feature', 'patch') == 'patch' else 577
            ) # might need to figure out how to do this more efficiently since the vision encoder might change...
        self.text_embedding_dim = getattr(training_config, 'mm_vision_output_text_embedding_size', 4096)
        self.kernel_size = getattr(training_config, 'mm_vision_token_compression_kernel_size', 4)
        self.stride = getattr(training_config, 'mm_vision_token_compression_stride', 4)

        # confirm we can convert the visual tokens to square grid
        assert (self.visual_token_count ** 0.5) % 1 == 0
        self.token_grid_size = int(self.visual_token_count ** 0.5)

        self.embed_dim = tower_config.hidden_size
        self.hidden_size = 4096
        self.num_heads = 8
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.clip_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True)

        self.q_proj = [nn.Linear(self.embed_dim, self.embed_dim)]
        self.q_proj.append(nn.GELU())
        self.q_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.q_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.q_proj = nn.Sequential(*self.q_proj)

        self.k_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.k_proj.append(nn.GELU())
        self.k_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.k_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.k_proj = nn.Sequential(*self.k_proj)

        self.v_proj = [nn.Linear(self.hidden_size, self.embed_dim)]
        self.v_proj.append(nn.GELU())
        self.v_proj.append(nn.Linear(self.embed_dim, self.embed_dim))
        self.v_proj.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.v_proj = nn.Sequential(*self.v_proj)

        self.q_downsample = torch.nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim,
                                            kernel_size=self.kernel_size, stride=self.stride,
                                            groups=self.embed_dim)
        
        self.text_projection = [nn.Linear(self.text_embedding_dim, self.embed_dim)]
        self.text_projection.append(nn.LayerNorm(self.embed_dim, eps=1e-6))
        self.text_projection = nn.Sequential(*self.text_projection)

    def forward(self, hidden_states):
        x = hidden_states[2] #orignal single level image features
        text_features = hidden_states[1]
        x_multi = hidden_states[0] #multi level image features

        # add in query now
        text_features = self.text_projection(text_features)
        x = x + text_features
        
        #add the query feature to the self
        query_states_2d = einops.rearrange(self.q_proj(x), 'b (h w) d -> b d h w',
                                            h = self.token_grid_size,
                                            w = self.token_grid_size)
        downsampled_q = self.q_downsample(query_states_2d) 
        b, _, h, w = downsampled_q.size()

        # makes it so each grid counts as a separate batch
        query_states = einops.rearrange(downsampled_q, 'b d h w -> (b h w) 1 d')

        key_states = self.k_proj(x_multi) # b x 576 x d
        value_states = self.v_proj(x_multi)

        # for "chunking" a 2d tensor into a 2d grid (a c) (b d) -> (a b) c d gives use a*b cxd tensors
        # e.g., setting a,b=2, allows use to segment the tensor into 4 quadrants
        k = self.token_grid_size // h
        l = self.token_grid_size // w
        key_states = einops.rearrange(key_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        value_states = einops.rearrange(value_states, "b (i k j l) d -> (b i j) (k l) d",
                         i=h, j=w, k=k, l=l)
        
        # attention is now from each convolution "grid" to the respective tokens
        attn_output = self.clip_attn(query_states, key_states, value_states)[0]

        output = einops.rearrange(attn_output, "(b t) 1 d -> b t d", b=b)

        return output