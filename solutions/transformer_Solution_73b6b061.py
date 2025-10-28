class EmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.ctx_len, config.d_model)
        self.device = config.device

    def forward(self, x):
        batch_size, seq_len = x.shape

        x_tok = self.wte(x)
        # print(x_tok.shape) uncomment this if you want to see the shape of above tensor
        x_pos = self.wpe(torch.arange(seq_len, device=self.device))
        # print(x_pos.shape)

        # Combine token and position embeddings
        x_embeddings = x_tok + x_pos

        return x_embeddings


# Testing your implementation
xb, yb = get_batch('train', config.ctx_len, config.batch_size, config.device)

embedding = EmbeddingLayer(config)
x_embedding = embedding(xb)

assert x_embedding.shape == (config.batch_size, config.ctx_len, config.d_model), "Embedding dimensions are incorrect"
print("Embedding layer output shape is correct!")