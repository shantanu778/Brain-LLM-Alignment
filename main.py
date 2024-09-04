import re
import os
import argparse
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from glob import glob
from datasets.dataset_dict import DatasetDict
from datasets import Dataset
import torchtext
from models import Custom_LSTM
from utils import get_data, create_vocab, get_batch

# def remove_non_alphabetic(title):
#     return re.sub('[^a-zA-Z]', ' ', title)



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
        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        #print(src.shape, target.shape)
        output, prediction, hidden = model(src, hidden)                  
        loss = criterion(prediction.reshape(batch_size*seq_len, -1) , target.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
        hidden = model.detach_hidden(hidden)
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in tqdm(range(0, num_batches - 1, seq_len)):
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            output, prediction, hidden = model(src, hidden)
            loss = criterion(prediction.reshape(batch_size*seq_len, -1) , target.reshape(-1))
            epoch_loss += loss.item() * seq_len
            hidden = model.detach_hidden(hidden)
    return epoch_loss / num_batches



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str,
                    help="Pass configuration json file")
 
    parser.add_argument('--test', default=False, action='store_true', help="Flag to do Test")
    args = parser.parse_args()

    f = open(args.file_name)

    config = json.load(f)

    # print(config['vocab'])
    
    
    batch_size = config['batch_size']
    random_seed = 42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    test = args.test
    c = []
    for file in glob(f'{config["corpus_type"]}/*_{config["corpus_type"]}.txt'):
        print(file)
        with open(file) as f:
            c.extend(f.readlines())


    #text = list(map(remove_non_alphabetic, c))

    data = Dataset.from_dict({'text':c})


    train_test = data.train_test_split(test_size=0.2, seed=random_seed)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=random_seed)

    d = DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
        })


    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  
    tokenized_dataset = d.map(tokenize_data, remove_columns=['text'], 
    fn_kwargs={'tokenizer': tokenizer})
    # print(tokenized_dataset['train'][:10])
    
    if test:
        print("Test")
        vocab = torch.load(f"{config['corpus_type']}/models/{config['vocab']}.pth")
    else:
        vocab = create_vocab(tokenized_dataset['train'])
        torch.save(vocab, f"{config['corpus_type']}/models/{config['vocab']}.pth")
    
    if test:
        test_data = get_data(tokenized_dataset['test'], vocab, batch_size)
    else:
        train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
        valid_data = get_data(tokenized_dataset['valid'], vocab, batch_size)
    
    #test_data = get_data(tokenized_dataset['test'], vocab, batch_size)

    vocab_size = len(vocab)
    embedding_dim = config["embedding_dim"]             # 400 in the paper
    hidden_dim = config["hidden_dim"]                # 1150 in the paper
    num_layers = config["num_layers"]                 # 3 in the paper
    dropout_rate = config["dropout_rate"]            
    tie_weights = config["tie_weights"]                  
    lr = config["lr"]

    model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')

    n_epochs = config["epochs"]
    seq_len = config["seq_len"]
    clip = config["clip"]
    

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    if test:
        print(f"{'='*20} Testing {'='*20}")
        model.load_state_dict(torch.load(f'{config["corpus_type"]}/models/{config["model"]}.pt',  map_location=device))
        test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {math.exp(test_loss):.3f}')
    else:
        best_valid_loss = float('inf')
        count = 0
        for epoch in range(n_epochs):
            print(f"Epoch: {epoch}")
            
            if count > 2:
                print("Early Stopped")
                break
            train_loss = train(model, train_data, optimizer, criterion, 
        	     batch_size, seq_len, clip, device)
            valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                     seq_len, device)

            lr_scheduler.step(valid_loss)

            if not os.path.exists(f'{config["corpus_type"]}/models/'):
                    os.makedirs(f'{config["corpus_type"]}/models/')
                    with open(f'{config["corpus_type"]}/models/config.json', 'w') as f:
                        json.dump(config, f)
                        f.close()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'{config["corpus_type"]}/models/{config["model"]}.pt')
                count = 0
                print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
                print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')
            else:
                count += 1

            #print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
            #print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')
        
        
