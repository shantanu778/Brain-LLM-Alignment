import argparse
import json
import torch
from torchtext import data
from models import Custom_LSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)

def generate_word(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

def generate_char(prompt, max_seq_len, temperature, model, encode, decode, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()

    indices = encode(prompt)
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            indices.append(prediction)

    # itos = vocab.get_itos()
    tokens = decode(indices)
    return tokens


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str,
                    help="Pass configuration json file")
 
    parser.add_argument("-t", "--text", type=str, help="Pass input text")
    parser.add_argument('--char', default=False, action='store_true', help="Flag to do character tokenization")


    args = parser.parse_args()

    f = open(args.file_name)

    config = json.load(f)

    if args.char:
        vocab = json.load(open(f'{config["corpus_type"]}/char/models/{config["vocab"]}.pth', "r"))
    else:
        vocab = torch.load(f'{config["corpus_type"]}/models/{config["vocab"]}.pth')

    max_seq_len = config['seq_len']
    seed = 0
    vocab_size = len(vocab)
    embedding_dim = config['embedding_dim']             # 400 in the paper
    hidden_dim = config['hidden_dim']                # 1150 in the paper
    num_layers = config['num_layers']                   # 3 in the paper
    dropout_rate = config['dropout_rate']
    tie_weights = config['tie_weights']
    lr = config['lr']
    # print(vocab)

    if args.char:
        str_to_ind =  vocab
        ind_to_str =  {i:ch for i, ch in enumerate(vocab)}
        encode = lambda s: [str_to_ind[i]for i in s]
        decode = lambda l: ''.join([ind_to_str[i] for i in l])
    else:
        tokenizer = data.utils.get_tokenizer('basic_english')

    model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)

    if args.char:
        model.load_state_dict(torch.load(f'{config["corpus_type"]}/char/models/{config["model"]}.pt',  map_location=device))
    else:
        model.load_state_dict(torch.load(f'{config["corpus_type"]}/models/{config["model"]}.pt',  map_location=device))

    prompt = args.text
    temperatures = [0.1, 0.2, 0.5, 0.7, 0.8, 1.0]
    for temperature in temperatures:
        if args.char:
            generation = generate_char(prompt, max_seq_len, temperature, model, encode, decode, device, seed)
            print(str(temperature)+'\n'+''.join(generation)+'\n')
        else:
            generation = generate_word(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
            print(str(temperature)+'\n'+' '.join(generation)+'\n')
        