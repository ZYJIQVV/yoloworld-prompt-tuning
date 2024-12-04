import torch
from torch import nn
import clip

class Prompter(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx):
        super(Prompter, self).__init__()
        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        dtype = clip_model.dtype
        # text prompt
        ctx_dim = clip_model.ln_final.weight.shape[0]
        ctx_vectors = torch.empty((n_ctx, ctx_dim), dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # vision prompt

    def forward(self, x):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,  # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        b, c, h, w = x[0].shape
        print()
        return prompts, x, self.tokenized_prompts

    # def to(self, *args, **kwargs):
    #     self = super().to(*args, **kwargs)
    #     self.token_prefix = self.token_prefix.to(*args, **kwargs)
    #     self.token_suffix = self.token_suffix.to(*args, **kwargs)
    #     self.tokenized_prompts = self.tokenized_prompts.to(*args, **kwargs)
    #     return self