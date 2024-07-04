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
        if example['tokens']:   
            # print(example['tokens'])                                   
            tokens = example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data


def get_char(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        if example['tokens']:   
            # print(example['text'])                                              
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data


def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+seq_len]             
    return src, target

