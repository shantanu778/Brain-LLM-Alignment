import os
import sys
import h5py
from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
import argparse
# Run this cell if your computer has a 'retina' or high DPI display. It will make the figures look much nicer.
import numpy as np
from npp import zscore
import logging
import pickle
import torch
import json
from dsutils import make_word_ds, make_phoneme_ds, make_semantic_model_gpt, make_semantic_model_gpt_word, make_semantic_model, make_semantic_model_word
from lstm.models import Custom_LSTM
from gpt2.main import GPTConfig, GPT
# Load TextGrids
from stimulus_utils import load_grids_for_stories
# Load TRfiles
from stimulus_utils import load_generic_trfiles
from util import make_delayed
# from IPython.display import HTML
logging.basicConfig(level=logging.DEBUG)

#%config InlineBackend.figure_format = 'retina'
print(torch.__version__)
dir_path = os.path.dirname(__file__)
print("Directory Path:", dir_path)

def project_stimuli(allstories, wordseqs, vocab_size, config):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    # Project stimuli
    torch.manual_seed(0)
    if config['model_type']=='lstm':
        embedding_dim = 768
        hidden_dim = 768
        num_layers = 2
        dropout_rate = 0.4
        tie_weights = False
        print("Loading LSTM Model")
        model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights)
        model.load_state_dict(torch.load(f'./{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt', map_location = device))
        model.to(device)
    else:
        print("Loading GPT2 Model")
        checkpoint = torch.load(f'./{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt', map_location = device)
        gptconf = checkpoint['config']
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        model.to(device)

    semanticseqs = dict() # dictionary to hold projected stimuli {story name : projected DataSequence}
    for story in allstories:
        # print(wordseqs[story].data)
        # f = open(f"plain/different_story/test_{len(wordseqs[story].data)}.txt", "r").read()
        # wordseqs[story].data = f.split(" ")
        # semanticseqs[story] = make_semantic_model(wordseqs[story], model, device)
        if config['model_type']=='lstm':
            semanticseqs[story] = make_semantic_model(wordseqs[story], model, device)
        else:
            semanticseqs[story] = make_semantic_model_gpt(wordseqs[story], model, device)

    return semanticseqs


# Make word and phoneme datasequences
def data_seq(allstories, vocab_file, config):
    # Here you will load the TextGrids for the stories, as well as 'TRfiles',
    # which specify the time points relative to story onset when the fMRI data was collected (roughly every 2 seconds).
    # Finally the TextGrids and TRfiles will be combined together into a representation I call a DataSequence. 
    # There is nothing interesting going on here scientifically, this is just something to make subsequent steps more manageable.

    flag = config['token_type'] # "word" or "char"
    grids = load_grids_for_stories(allstories, "alignment/data/grids")
    trfiles = load_generic_trfiles(allstories, "alignment/data/trfiles")
    wordseqs = make_word_ds(vocab_file, flag, grids, trfiles) # dictionary of {storyname : word DataSequence}
    return wordseqs

def llm_representation(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str,
                    help="Pass configuration json file")
    
    parser.add_argument('--sub', type=str, required=True, help="Subject ID")
    parser.add_argument('--roi', type=str, required=True, help="Region of Interest")
    parser.add_argument('--test', default=False, action='store_true', help="Flag to do Test")
    parser.add_argument('--plain', default=False, action='store_true', help="Flag to Use Linear Model trained on plain text")

    args = parser.parse_args()

    f = open(args.file_name)

    config = json.load(f)
    print(config)
    meta_path = f'./{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["vocab"]}.pkl'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    vocab_size = meta['vocab_size']

    # These are lists of the stories
    # Rstories are the names of the training (or Regression) stories, which we will use to fit our models
    Rstories = ["alternateithicatom", "howtodraw", "sloth", "naked", "souls", "avatar", "legacy", "myfirstdaywiththeyankees", "odetostepfather", "undertheinfluence"]
    
    # Rstories = ['stagefright']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']

    allstories = Rstories + Pstories
    
    wordseqs = data_seq(allstories, meta_path, config)

    semanticseqs = project_stimuli(allstories, wordseqs, vocab_size, config)

    # take a look at the projected stimuli
    # naked_proj = semanticseqs["naked"]

    # print (naked_proj.data.shape) # prints the shape of 'data' as (rows, columns)
    # print (naked_proj.data[:10]) # print the first 10 rows (this will be truncated)
    # Downsample stimuli
    interptype = "lanczos" # filter type
    window = 3 # number of lobes in Lanczos filter

    downsampled_semanticseqs = dict() # dictionary to hold downsampled stimuli
    for story in allstories:
        downsampled_semanticseqs[story] = semanticseqs[story].chunksums(interptype, window=window)

    # Combine stimuli
    trim = 5
    Rstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories])
    Pstim = np.vstack([zscore(downsampled_semanticseqs[story][5+trim:-trim]) for story in Pstories])

    storylens = [len(downsampled_semanticseqs[story][5+trim:-trim]) for story in Rstories]
    print(storylens)

    print(np.cumsum(storylens))
    # Delay stimuli
   
    ndelays = 4
    delays = range(1, ndelays+1)

    print ("FIR model delays: ", delays)

    delRstim = make_delayed(Rstim, delays)
    delPstim = make_delayed(Pstim, delays)
    # Print the sizes of these matrices
    print ("delRstim shape: ", delRstim.shape)
    print ("delPstim shape: ", delPstim.shape)

    # save delayed llm representations
    hdf5_filepath = Path(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/stories_rep.hf5')
    hdf5_filepath.parent.mkdir(parents=True, exist_ok=True)


    with h5py.File(hdf5_filepath, 'w') as hf:
        hf.create_dataset("delRstim",  data=delRstim, dtype='f')
        hf.create_dataset("delPstim",  data=delPstim, dtype='f')


if __name__=='__main__':
    llm_representation()