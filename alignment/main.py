import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import torch
# Run this cell if your computer has a 'retina' or high DPI display. It will make the figures look much nicer.
from matplotlib.pyplot import figure, cm
import numpy as np
from npp import zscore
import logging
import tables
import pickle
import cortex
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
from ridge import bootstrap_ridge
import cortex
from IPython.display import HTML
logging.basicConfig(level=logging.DEBUG)

#%config InlineBackend.figure_format = 'retina'
print(torch.__version__)
dir_path = os.path.dirname(__file__)
print(dir_path)

def project_stimuli(allstories, wordseqs, vocab, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Project stimuli
    torch.manual_seed(0)
    if config['model_type']=='lstm':
        vocab_size = len(vocab)
        embedding_dim = 768
        hidden_dim = 768
        num_layers = 2
        dropout_rate = 0.4
        tie_weights = False
        print("Loading LSTM Model")
        model = Custom_LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
        model.load_state_dict(torch.load(f'./{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt'))
    else:
        print("Loading GPT2 Model")
        checkpoint = torch.load(f'./{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/{config["model"]}.pt')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str,
                    help="Pass configuration json file")
 
    parser.add_argument('--test', default=False, action='store_true', help="Flag to do Test")

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
    Rstories = ['alternateithicatom', 'avatar', 'howtodraw', 'legacy',
                'life', 'myfirstdaywiththeyankees', 'naked',
                'odetostepfather', 'souls', 'undertheinfluence']

    # Pstories are the test (or Prediction) stories (well, story), which we will use to test our models
    Pstories = ['wheretheressmoke']

    allstories = Rstories + Pstories
    
    wordseqs = data_seq(allstories, meta_path, config)

    semanticseqs = project_stimuli(allstories, wordseqs, vocab_size, config)

    # take a look at the projected stimuli
    naked_proj = semanticseqs["naked"]

    print (naked_proj.data.shape) # prints the shape of 'data' as (rows, columns)
    print (naked_proj.data[:10]) # print the first 10 rows (this will be truncated)
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

    # Load responses
    resptf = tables.open_file("alignment/data/fmri-responses.hf5")
    zRresp = resptf.root.zRresp.read()
    zPresp = resptf.root.zPresp.read()
    mask = resptf.root.mask.read()

    # Run regression
    alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 5 # Number of cross-validation runs.
    chunklen = 40 #
    nchunks = 20


    wt, corrs, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                        alphas, nboots, chunklen, nchunks,
                                                        singcutoff=1e-10, single_alpha=True)
    
    # with open("wt_gpt_global.pkl", "wb") as f:
    # pickle.dump(wt, f)

    # wt is the regression weights
    print ("wt has shape: ", wt.shape)
    # corr is the correlation between predicted and actual voxel responses in the Prediction dataset
    print ("corr has shape: ", corrs.shape)
    # alphas is the selected alpha value for each voxel, here it should be the same across voxels
    print ("alphas has shape: ", alphas.shape)
    # bscorrs is the correlation between predicted and actual voxel responses for each round of cross-validation
    # within the Regression dataset
    print ("bscorrs has shape (num alphas, num voxels, nboots): ", bscorrs.shape)
    # valinds is the indices of the time points in the Regression dataset that were used for each
    # round of cross-validation
    print ("valinds has shape: ", np.array(valinds).shape)

    # Predict responses in the Prediction dataset

    # First let's refresh ourselves on the shapes of these matrices
    print ("zPresp has shape: ", zPresp.shape)
    print ("wt has shape: ", wt.shape)
    print ("delPstim has shape: ", delPstim.shape)

    # Then let's predict responses by taking the dot product of the weights and stim
    pred = np.dot(delPstim, wt)

    print ("pred has shape: ", pred.shape)
    # nnpred = np.nan_to_num(pred)
    # corrs = np.nan_to_num(np.array([np.corrcoef(zPresp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(zPresp.shape[1])]))

    selvox = 20710 # a decent voxel
    # Compute correlation between single predicted and actual response
    # (np.corrcoef returns a correlation matrix; pull out the element [0,1] to get
    # correlation between the two vectors)
    voxcorr = np.corrcoef(zPresp[:,selvox], pred[:,selvox])[0,1]
    print ("Correlation between predicted and actual responses for voxel %d: %f" % (selvox, voxcorr))

    voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
    for vi in range(zPresp.shape[1]):
        voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]

    # Plot histogram of correlations
    f = figure(figsize=(8,8))
    ax = f.add_subplot(1,1,1)
    ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Num. voxels");

    # @title
    # Plot mosaic of correlations
    corrvolume = np.zeros(mask.shape)
    corrvolume[mask>0] = voxcorrs

    f = figure(figsize=(10,10))
    cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot)

    # Plot correlations on cortex
    corrvol = cortex.Volume(corrs, "S1", "fullhead", mask=mask, vmin=0, vmax=0.5, cmap='hot')
    cortex.webshow(corrvol, port=8881, open_browser=False)

    # View 3D model
    # You will need to change where it says SERVERIP below to the IP you are connected to
    
    HTML("<a target='_blank' href='https://127.0.0.1/:8888'>Click here for viewer</a>")
    # Plot correlation flatmap
    cortex.quickshow(corrvol, with_rois=False, with_labels=False)


if __name__ == '__main__':
    main()