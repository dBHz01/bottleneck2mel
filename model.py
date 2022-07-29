import torch
import torch.nn as nn

BOTTLE_NECK_SIZE = 256
BATCH_SIZE = 16


class Bottleneck2MelModel(nn.Module):
    def __init__(self, field_dim, embedding_dim, head_num, output_size, p_drop=0.3, attn_num=6):
        super(Bottleneck2MelModel, self).__init__()
        self.embedding = torch.nn.Embedding(
            field_dim, embedding_dim)  # embed speaker id
        self.drop = nn.Dropout(p_drop)
        self.multihead_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(BOTTLE_NECK_SIZE + embedding_dim, head_num, dropout=p_drop, batch_first=True) for _ in range(attn_num)
        ])
        self.norm = nn.LayerNorm(BOTTLE_NECK_SIZE + embedding_dim)
        self.linear = nn.Linear(BOTTLE_NECK_SIZE + embedding_dim, output_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs, speaker_id):
        inputs = torch.repeat_interleave(inputs, 2, dim=1)  # double its frames
        embeded = self.embedding(speaker_id)
        embeded = embeded.unsqueeze(1)
        embeded = embeded.repeat(1, inputs.shape[1], 1)
        # print(inputs.shape)  # torch.Size([128(batch_size), 522(frames), 256])
        # print(embeded.shape) # torch.Size([128(batch_size), 522(frames), 128])

        embeded_input = torch.concat((inputs, embeded), 2)
        # print(embeded_input.shape) # # torch.Size([128(batch_size), 522(frames), 384])

        for attn in self.multihead_attns:
            inp_atten = attn(embeded_input, embeded_input, embeded_input)[0]
            embeded_input = self.norm(inp_atten + embeded_input)

        return self.linear(embeded_input).squeeze()


class Bottleneck2MelModel2(nn.Module):
    def __init__(self, field_dim, embedding_dim, head_num, output_size, p_drop=0.3, attn_num=6):
        super(Bottleneck2MelModel2, self).__init__()
        self.embedding = torch.nn.Embedding(
            field_dim, embedding_dim)  # embed speaker id
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=BOTTLE_NECK_SIZE * 2, nhead=head_num, batch_first=True, dropout=p_drop)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=attn_num)
        self.linear_1 = nn.Linear(
            BOTTLE_NECK_SIZE + embedding_dim, BOTTLE_NECK_SIZE * 2)
        self.linear_2 = nn.Linear(BOTTLE_NECK_SIZE * 2, output_size)
        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.linear_1.bias.data.zero_()
        self.linear_1.weight.data.uniform_(-init_range, init_range)
        self.linear_2.bias.data.zero_()
        self.linear_2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inputs, speaker_id):
        inputs = torch.repeat_interleave(inputs, 2, dim=1)  # double its frames
        embeded = self.embedding(speaker_id)
        embeded = embeded.unsqueeze(1)
        embeded = embeded.repeat(1, inputs.shape[1], 1)
        # print(inputs.shape)  # torch.Size([128(batch_size), 522(frames), 256])
        # print(embeded.shape) # torch.Size([128(batch_size), 522(frames), 128])

        embeded_input = torch.concat((inputs, embeded), 2)
        # print(embeded_input.shape) # # torch.Size([128(batch_size), 522(frames), 384])

        # torch.Size([128(batch_size), 522(frames), 512])
        embeded_input = self.linear_1(embeded_input)

        embeded_input = self.encoder(embeded_input)

        return self.linear_2(embeded_input).squeeze()
