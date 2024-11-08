import os
import json
import argparse
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets.dataset_dict import DatasetDict
import math
import datasets
import torchtext
from tqdm import tqdm
from utils import get_char, get_data, get_batch, create_vocab
# from main import train, evaluate
from models import Custom_LSTM

# Check if CUDA is available (i.e., if at least one GPU is available)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
print(f"Torch Version: {torch.__version__}, Device: {device}")

def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device):
    
    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):  # The last batch can't be a src
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)
        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        #print(src.shape, target.shape)
        output, prediction, hidden = model(src, hidden)                  
        loss = criterion(prediction.reshape(batch_size * seq_len, -1), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Validation: ',leave=False):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            output, prediction, hidden = model(src, hidden)
            loss = criterion(prediction.reshape(batch_size*seq_len, -1) , target.reshape(-1))
            epoch_loss += loss.item() * seq_len
            
    return epoch_loss / num_batches

# class TextDataset(Dataset):
#     def __init__(self, text, sequence_length):
#         self.text = text
#         self.sequence_length = sequence_length
#         self.text_length = len(text) - sequence_length

#     def __len__(self):
#         return self.text_length

#     def __getitem__(self, idx):
#         seq = self.text[idx: idx + self.sequence_length]
#         next_char = self.text[idx + self.sequence_length]
#         return torch.tensor(seq['tokens'], dtype=torch.long), torch.tensor(next_char['tokens'], dtype=torch.long)


# class LSTMModel(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout_rate):
#         super(LSTMModel, self).__init__()
#         self.num_layers = n_layers
#         self.hidden_dim = hidden_dim
#         self.embedding_dim = embedding_dim
#         self.dropout = nn.Dropout(dropout_rate)
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout_rate, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#     def forward(self, x, hidden):
#         x = self.embedding(x)
#         out, hidden = self.lstm(x, hidden)
#         out = self.dropout(out)
#         out = self.fc(out[:,-1])
#         return out, hidden

#     def init_hidden(self, batch_size, device):
#         hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
#         cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
#         return hidden, cell

#     def detach_hidden(self, hidden):
#         hidden, cell = hidden
#         hidden = hidden.detach()
#         cell = cell.detach()
#         return hidden, cell



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str,
                    help="Pass configuration json file")
 
    parser.add_argument('--test', default=False, action='store_true', help="Flag to do Test")
    parser.add_argument('--char', default=False, action='store_true', help="Flag to Train Word Based Model")

    args = parser.parse_args()

    f = open(args.file_name)

    config = json.load(f)

    if not os.path.exists(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/'):
        os.makedirs(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/')
    with open(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/config.json', 'w') as f:
        json.dump(config, f)
        f.close()
    batch_size = config['batch_size']
    random_seed = 42
    test = args.test

    text = []
    for file in glob(f'{config["corpus_type"]}/*_{config["corpus_type"]}.txt'):
        print(file)
        with open(file) as f:
            text.extend(f.readlines())

    

    d = datasets.Dataset.from_dict({'text':text})

    random_seed = 42
    train_test = d.train_test_split(test_size=0.2, shuffle=False, seed=random_seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, shuffle=False, seed=random_seed)

    d = DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
        })

    if not args.char:
        print(f'{10*"="} Applying Word based Tokenization {10*"="}')
        tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  
        tokenized_dataset = d.map(tokenize_data, remove_columns=['text'], 
        fn_kwargs={'tokenizer': tokenizer})
        if test:
            print("Test")
            vocab = torch.load(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["vocab"]}.pth')
            test_data = get_data(tokenized_dataset['test'], vocab, batch_size)
        else:
            vocab = create_vocab(tokenized_dataset['train'])
            print("Saving Vocabulary")
            torch.save(vocab, f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["vocab"]}.pth')
            train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
            valid_data = get_data(tokenized_dataset['valid'], vocab, batch_size)
        # print(d['train'][:10])
    else:
        print(f'{10*"="} Applying Character based Tokenization {10*"="}')
        vocab = sorted(set(" ".join(text)))
        vocab_size=len(vocab)
        str_to_ind =  {ch:i for i, ch in enumerate(vocab)}
        ind_to_str =  {i:ch for i, ch in enumerate(vocab)}
        encode = lambda example: {'tokens': list(str_to_ind[c] for c in example['text'])}
        decode = lambda l: ''.join([ind_to_str[i] for i in l])
        # tokenized_dataset = d.map(encode, remove_columns=['text'], batched=True)
        # print(d['test'])
        if test:
            test_data = get_char(d["test"], str_to_ind, batch_size)
        else:
            train_data = get_char(d['train'], str_to_ind, batch_size)
            valid_data = get_char(d['valid'], str_to_ind, batch_size)
    
    
        with open(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["vocab"]}.pth', 'w') as f:
            json.dump(vocab, f)
            f.close()

    vocab_size = len(vocab)
    embedding_dim = config["embedding_dim"]             # 400 in the paper
    hidden_dim = config["hidden_dim"]                # 1150 in the paper
    num_layers = config["num_layers"]                 # 3 in the paper
    dropout_rate = config["dropout_rate"]            
    tie_weights = config["tie_weights"]                  
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    n_epochs = config["epochs"]
    seq_len = config["seq_len"]
    clip = config["clip"]


    print("Loading LSTM Model")
    model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    # model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    
    if test:
        print(f"{'='*20} Testing {'='*20}")
        model.load_state_dict(torch.load(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt',  map_location=device))
        test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {math.exp(test_loss):.3f}')
    else:
        best_valid_loss = float('inf')
        count = 0
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")
            
            if count > 3:
                print("Early Stopped")
                break
            train_loss = train(model, train_data, optimizer, criterion, 
        	     batch_size, seq_len, clip, device)
            valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                     seq_len, device)
    
            lr_scheduler.step(valid_loss)
            print(f'\t\t Validation Loss: {valid_loss:.3f}')
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt')
                count = 0
                print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
                print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')
            
            else:
                count += 1    


if __name__ == '__main__':
    main()
