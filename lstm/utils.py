import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from glob import glob
import torchtext
import torch

def create_vocab(tokenized_dataset):
    vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['tokens'], min_freq=5) 
    vocab.insert_token('<unk>', 0)           
    vocab.insert_token('<eos>', 1)            
    vocab.set_default_index(vocab['<unk>'])   
    print(len(vocab))                         
    # print(vocab.get_itos()[:10])   
    return vocab

def get_data(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        # print(example)
        if example['tokens']:   
            # print(example['tokens'])                                   
            example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data

def get_data_bpe(dataset, batch_size):
    data = []                                                   
    for example in dataset:
        print(example)
        if example['input_ids']:   
            # print(example['tokens'])                                   
            # example['tokens'].append('<eos>')             
            # tokens = [vocab[token] for token in example['tokens']] 
            data.extend(example['input_ids'])                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data


def get_char(dataset, vocab, batch_size):
    # print(vocab)
    data = []                                                   
    for example in dataset:
        # print(example['text'])
        if example['text']:   
            # print(example['text'])  
            # example['text'] += ' <eos>'                                             
            tokens = [vocab[token] for token in example['text']] 
            data.extend(tokens)  
    # print(dataset['tokens'])                                  
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)        
    return data


def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

