import torch

def total():
    beam_size = 10
    trg_len = 10
    for t in range(1, trg_len+1):
        output, hidden, cell = decoder()
        if t == 1:
            outputs = find_max_output_by_beam_search(output, beam_size)
        else:
            outputs = beam_search_with_decoder(outputs, beam_size)
            outputs = compute_beam_search_elements_prob(outputs)
            outputs = remove_beam_search_elements(outputs, beam_size)
    return outputs

def decoder():
    output = torch.rand(10)
    hidden = torch.Tensor([0.5,0.4,0.3])
    cell = torch.Tensor([0.2,0.8,0.2])
    return output, hidden, cell

def find_max_output_by_beam_search(output, beam_size):
    outputs = []
    for _ in range(beam_size):
        max_value, max_index = output.max(0)
        output[max_index] = 0.0
        outputs.append([max_index, max_value])
    return outputs

def beam_search_with_decoder(outputs, beam_size):
    for i in range(beam_size):
        output, _, _ = decoder()
        outputs[i].append(find_max_output_by_beam_search(output, beam_size)) #### 이부분에 디코더 들어가야함
    return outputs

def compute_beam_search_elements_prob(outputs):
    for i in range(len(outputs)):
        prob = outputs[i][-2]
        for j in range(len(outputs[i][-1])):
            outputs[i][-1][j][-1] = outputs[i][-1][j][-1]*prob
    return outputs

def remove_beam_search_elements(outputs, beam_size, max_value=None):
    '''
    간소화할 수 있음. 해야함.
    '''
    max_value = torch.zeros(beam_size)
    for i in range(len(outputs)):
        for j in range(len(outputs[i][-1])):
            sentence_prob = outputs[i][1]*outputs[i][-1][j][1]
            temp_max_value, temp_max_index = max_value.min(0)
            if sentence_prob > temp_max_value:
                max_value[temp_max_index] = sentence_prob

    save_words_index = [[torch.Tensor([]), torch.Tensor([])] for _ in range(beam_size)]
    for i in range(len(outputs)):
        for j in range(len(outputs[i][-1])):
            temp_max_value, temp_max_index = max_value.min(0)
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

if __name__ == '__main__':
    result = total()
    print(result)