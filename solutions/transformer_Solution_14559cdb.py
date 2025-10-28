class TokenEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape

        x_tok = self.wte(x)

        return x_tok


# Test your implementation
xb, yb = get_batch('train', config.ctx_len, config.batch_size, config.device)

token_embedding = TokenEmbeddingLayer(config)
x_tok = token_embedding(xb)

assert x_tok.shape == (config.batch_size, config.ctx_len, config.d_model), "Embedding dimensions are incorrect"
print("Token embedding layer output shape is correct!")