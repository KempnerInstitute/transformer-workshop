class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Stack of decoder blocks
        self.blocks = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layers)])

        # Final layer norm (normalize across d_model dimension)
        self.ln = nn.LayerNorm(config.d_model)

        # Linear projection from d_model to vocab_size
        self.lin = nn.Linear(config.d_model, config.vocab_size)

        # Embeddings
        self.emb = EmbeddingLayer(config)

        # Loss function for training
        self.L = nn.CrossEntropyLoss()
        self.ctx_len = config.ctx_len

        self.device = config.device # don't change this (for training model on right device)

    def forward(self, x, targets=None):
        """
        Args:
            x: Input tokens (B, T)
            targets: Optional target tokens (B, T)
        Returns:
            logits: Predictions (B, T, vocab_size)
            loss: Optional cross-entropy loss
        """
        batch_size, seq_len = x.shape

        # Embed tokens (token + positional embeddings)
        x = self.emb(x)

        # Process through the stack of transformer blocks
        x = self.blocks(x)

        # Apply final layer normalization
        x = self.ln(x)

        # Project from hidden dimension to vocabulary size
        logits = self.lin(x)

        # Compute loss if targets are provided
        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size*seq_len, vocab_size)
            targets = targets.view(batch_size*seq_len)

            # Compute loss
            loss = self.L(logits, targets)

        return logits, loss

    def generate(self, token_ids, max_len=256):
        """
        Generate new tokens given initial sequence of token IDs.

        Args:
            token_ids (torch.Tensor):
                The starting sequence of token IDs, shape (batch_size, seq_len).
            max_len (int, optional):
                Maximum number of new tokens to generate.
                Defaults to 256.

        Returns:
            torch.Tensor:
                The complete sequence of generated token IDs, including both the
                original input and the newly generated tokens.
                Shape: (batch_size, seq_len + max_len).
        """

        for _ in range(max_len):

            # Grab the last ctx_len tokens
            token_window = token_ids[:, -self.ctx_len:]

            # Get model predictions
            logits, _ = self(token_window)

            # Only keep predictions for the last token
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token (hint, can use torch.multinomial)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append next token to the sequence
            token_ids = torch.cat((token_ids, next_token), dim=1)

        return token_ids


# Testing your implementation
config = Config(
    vocab_size=100,
    d_model=256,
    ctx_len=64,
    n_layers=4
)
decoder = Decoder(config)

x = torch.randint(0, 100, (1, 10))
logits, loss = decoder(x, x)

out = decoder.generate(torch.tensor([[1, 2, 3]]), max_len=5)
out = decoder.generate(torch.tensor([[1, 2, 3]]), max_len=5)
assert out.shape == (1, 8)
print("Decoder output shape is correct!")
