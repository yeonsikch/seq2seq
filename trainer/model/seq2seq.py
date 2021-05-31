import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers)

    def forward(self, src):
        embedded = self.embedding(src)
        _, (hidden, cell) = self.rnn(embedded) # '_'는 원래 'output'인데 인코더에서는 사용하지 않음.
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers=4):
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)  # shape 확인
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.softmax(self.fc_out(output.squeeze(0)))  # shape 확인
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hidden_dim == decoder.hidden_dim
        assert encoder.n_layers == decoder.n_layers

    def forward(self, src, trg):

        batch_size = trg.shape[0]  # batch_first = True
        trg_len = trg.shape[1]  # batch_first = True
        trg_vocab_size = self.decoder.ouput_dim

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)  # batch_first=True여도 이렇다.
        hidden, cell = self.encoder(src)

        # decoder의 첫 input은 <SOS>이다.
        input = trg[0, :]

        # 마찬가지로, 1부터 시작하는 이유는 0은 <SOS>이기 때문.
        # target length만큼 반복
        for t in range(1, trg_len):
            # encoder의 output인 hidden과 cell을 decoder의 input으로 넣음.
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            input = output.argmax(1)

        return outputs

    '''
    beam_search 구현해야함. 논문에도 이거하면 성능 더 좋아졌다고 함.
    '''
    def beam_search(self, output, beam_size):
        outputs = []
        for _ in beam_size:
            max_value, max_index = output.max(0)
            output[max_index] = 0.0
            outputs.append([max_index, max_value])

