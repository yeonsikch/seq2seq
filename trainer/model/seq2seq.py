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
        input = input.unsqueeze(0)  # (embedding) -> (batch, embedding) 정확하게는 실제 shape을 확인해봐야할 듯.
        embedded = self.embedding(input)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.softmax(self.fc_out(output.squeeze(0)))  # 결과를 input으로 다시 넣아야하니 (batch, embedding) -> (embedding) 정확하게는 실제 shape을 확인해봐야할 듯.
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
        input = trg[0, :] # embedding의 0번 째 벡터 즉, <SOS> 토큰의 임베딩 값

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
    구현중 구현중 구현중 구현중 구현중 구현중 구현중 구현중 구현중
    '''

    '''
    def total():
        beam_size = 4
        trg_len = 5
        for t in range(1, trg_len):
            output, hidden, cell = decoder(torch.random.seed())
            if t == 1:
                outputs = find_max_output_by_beam_search(output, beam_size)
            else:
                outputs = beam_search_with_decoder(outputs, beam_size)
                outputs = compute_beam_search_elements_prob(outputs)
                outputs = remove_beam_search_elements(outputs)
        return outputs

    def decoder(seed):
        torch.random.manual_seed(seed)
        output = torch.rand(9)
        hidden = torch.Tensor([0.5, 0.4, 0.3])
        cell = torch.Tensor([0.2, 0.8, 0.2])
        return output, hidden, cell

    def find_max_output_by_beam_search(output, beam_size):
        outputs = []
        for _ in range(beam_size):
            max_value, max_index = output.max(0)
            output[max_index] = 0.0
            outputs.append([max_index, max_value])
        return outputs

    def beam_search_with_decoder(outputs, beam_size):
        # outputs = find_max_output_by_beam_search(output, beam_size)
        if len(outputs) == beam_size:
            output, _, _ = decoder(torch.random.seed())
            for i in range(beam_size):
                outputs[i].append(find_max_output_by_beam_search(output, beam_size))  #### 이부분에 디코더 들어가야함
        return outputs

    def compute_beam_search_elements_prob(outputs):
        for i in range(len(outputs)):
            prob = outputs[i][-2]
            for j in range(len(outputs[i][-1])):
                outputs[i][-1][j][-1] = outputs[i][-1][j][-1] * prob
        return outputs

    def remove_beam_search_elements(outputs, max_value=None, max_index=None):
        if max_value or max_index == None:
            max_value = torch.zeros(4)
            max_index = torch.zeros(4)
        save_words_index = [[torch.Tensor([]), torch.Tensor([])] for _ in range(4)]
        for i in range(len(outputs)):
            for j in range(len(outputs[i][-1])):
                temp_max_value, temp_max_index = max_value.min(0)
                # save_words_index = [[torch.Tensor([]),torch.Tensor([])] for _ in range(4)]
                sentence_prob = outputs[i][-1][j][-1]
                if sentence_prob > temp_max_value:
                    max_value[temp_max_index] = outputs[i][-1][j][-1]
                    previous_words = outputs[i][0]
                    try:
                        len(previous_words)
                    except:
                        previous_words = previous_words.unsqueeze(0)
                    new_word = outputs[i][-1][j][0]
                    new_word = new_word.unsqueeze(0)
                    total_words = torch.cat([previous_words, new_word])
                    save_words_index[temp_max_index][0] = total_words
                    save_words_index[temp_max_index][1] = sentence_prob
        return save_words_index
    '''