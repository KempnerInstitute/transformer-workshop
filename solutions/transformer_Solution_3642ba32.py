class SingleHeadCausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.d_model // config.n_heads
        self.key = nn.Linear(config.d_model, self.head_dim, bias=False)
        self.query = nn.Linear(config.d_model, self.head_dim, bias=False)
        self.value = nn.Linear(config.d_model, self.head_dim, bias=False)

        self.register_buffer("cmask", torch.tril(torch.ones([config.ctx_len, config.ctx_len])))


    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Step 1: Compute K, Q, V projections
        K = self.key(x) # (batch_size, seq_len, head_dim)
        Q = self.query(x) # (batch_size, seq_len, head_dim)
        V = self.value(x) # (batch_size, seq_len, head_dim)

        # Step 2: Compute scaled attention scores
        attention_scores = Q @ K.transpose(-2, -1) * self.head_dim**-0.5  # (batch_size, seq_len, seq_len)

        masked_scores = torch.masked_fill(attention_scores, self.cmask[:seq_len, :seq_len]==0, float('-inf'))
        attention_weights = F.softmax(masked_scores, dim=-1)
        outputs = attention_weights @ V
        return outputs


# Test your implementation
config = Config(d_model=256, n_heads=8, ctx_len=16)
attention = SingleHeadCausalAttention(config)
x = torch.randn(2, 10, 256)  # (batch_size, seq_len, d_model)
output = attention(x)
assert output.shape == (2, 10, 32)  # head_dim = 256/8 = 32
print("Single-head causal attention output shape is correct!")