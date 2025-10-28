class MultiHeadCausalAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadCausalAttention(config) for _ in range(config.n_heads)])

        self.linear = nn.Linear(config.d_model, config.d_model)


    def forward(self, x):
        y = torch.cat([h(x) for h in self.heads], dim=-1)
        y = self.linear(y)
        return y


# Testing your implementation
config = Config(d_model=256, n_heads=8, ctx_len=16)
mha = MultiHeadCausalAttention(config)

# Test with small batch
x = torch.randn(2, 10, 256)  # (batch_size=2, seq_len=10, d_model=256)
out = mha(x)
assert out.shape == (2, 10, 256)