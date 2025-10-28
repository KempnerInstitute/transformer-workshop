class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadCausalAttention(config)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ffn = FFN(config)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# Test your implementation
config = Config(d_model=256)
decoder = DecoderBlock(config)

# Test with random input
x = torch.randn(2, 10, 256)  # (batch_size, sequence_length, d_model)
output = decoder(x)
assert output.shape == x.shape
print("Decoder block output shape is correct!")