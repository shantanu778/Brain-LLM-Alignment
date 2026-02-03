import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import torch
# Run this cell if your computer has a 'retina' or high DPI display. It will make the figures look much nicer.
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
# from IPython.display import HTML
logging.basicConfig(level=logging.DEBUG)

#%config InlineBackend.figure_format = 'retina'
print(torch.__version__)
dir_path = os.path.dirname(__file__)
print(dir_path)

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

def main():
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

    # Load responses
    print(f"Loading responses from subject {args.sub} and ROI {args.roi}")
    resptf = tables.open_file(f"alignment/data/sub-{args.sub}/sub-{args.sub}_{args.roi}.hf5")

    zRresp = resptf.root.zRresp.read()
    zPresp = resptf.root.zPresp.read()
    mask = resptf.root.mask.read()
    print ("zRresp shape: ", zRresp.shape)
    print ("zPresp shape: ", zPresp.shape)  
    print ("mask shape: ", mask.shape)

    # Run regression
    alphas = np.logspace(1, 3, 10) # Equally log-spaced alphas between 10 and 1000. The third number is the number of alphas to test.
    nboots = 5 # Number of cross-validation runs.
    chunklen = 40 #
    nchunks = 20
    
    if args.test and not args.plain:
        print(f'Loading Plain Linear Model')
        wt = pickle.load(open(f'{config["corpus_type"]}/wt_{args.sub}_{args.roi}_{config["model_type"]}.pkl', "rb"))
        pred = np.dot(delPstim, wt)
        nnpred = np.nan_to_num(pred)
        corrs = np.nan_to_num(np.array([np.corrcoef(zPresp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(zPresp.shape[1])]))

    elif args.plain:
        print(f'Loading Plain Linear Model')
        wt = pickle.load(open(f'{config["corpus_type"]}/wt_{args.sub}_{args.roi}_{config["model_type"]}.pkl', "rb"))
        pred = np.dot(delPstim, wt)
        nnpred = np.nan_to_num(pred)
        corrs = np.nan_to_num(np.array([np.corrcoef(zPresp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(zPresp.shape[1])]))
    else:
        wt, corrs, alphas, bscorrs, valinds = bootstrap_ridge(delRstim, zRresp, delPstim, zPresp,
                                                            alphas, nboots, chunklen, nchunks,
                                                            singcutoff=1e-10, single_alpha=True)
        
        os.makedirs(f'{config["corpus_type"]}/sub-{args.sub}', exist_ok=True)
        with open(f'{config["corpus_type"]}/sub-{args.sub}/wt_{args.sub}_{args.roi}_{config["model_type"]}.pkl', "wb") as f:
            pickle.dump(wt, f)


        pred = np.dot(delPstim, wt)
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
    # corrs = np.nan_to_num(np.array([np.corrcoef(zPresp[:,ii], nnpred[:,ii].ravel())[0,1] for ii in range(zPresp.shape[1])]))

    print ("pred has shape: ", pred.shape)
    # nnpred = np.nan_to_num(pred)

    if args.plain:
        with open(f'{config["corpus_type"]}/sub-{args.sub}/corrs_{args.sub}_{args.roi}_plain.npy', 'wb') as f:
            np.save(f, corrs)
    else:
        with open(f'{config["corpus_type"]}/sub-{args.sub}/corrs_{args.sub}_{args.roi}_{config["model_type"]}.npy', 'wb') as f:
            np.save(f, corrs)

    
    # selvox = 20710 # a decent voxel
    # Compute correlation between single predicted and actual response
    # (np.corrcoef returns a correlation matrix; pull out the element [0,1] to get
    # correlation between the two vectors)
    # voxcorr = np.corrcoef(zPresp[:,selvox], pred[:,selvox])[0,1]
    # print ("Correlation between predicted and actual responses for voxel %d: %f" % (selvox, voxcorr))

    # voxcorrs = np.zeros((zPresp.shape[1],)) # create zero-filled array to hold correlations
    # for vi in range(zPresp.shape[1]):
    #     voxcorrs[vi] = np.corrcoef(zPresp[:,vi], pred[:,vi])[0,1]

    # # Plot histogram of correlations
    # f = figure(figsize=(8,8))
    # ax = f.add_subplot(1,1,1)
    # ax.hist(voxcorrs, 100) # histogram correlations with 100 bins
    # ax.set_xlabel("Correlation")
    # ax.set_ylabel("Num. voxels")

    # # @title
    # # Plot mosaic of correlations
    # corrvolume = np.zeros(mask.shape)
    # corrvolume[mask>0] = voxcorrs

    # f = figure(figsize=(10,10))
    # cortex.mosaic(corrvolume, vmin=0, vmax=0.5, cmap=cm.hot)

    # Plot correlations on cortex
    # corrvol = cortex.Volume(corrs, "S1", "fullhead", mask=mask, vmin=0, vmax=0.5, cmap='hot')
    # cortex.add_roi(corrvol, name = 'test', add_path=True, open_inkscape = True)
    # cortex.webshow(corrvol, port=8888, open_browser=False, recache = True)

    # # View 3D model
    # # You will need to change where it says SERVERIP below to the IP you are connected to

    # subject = "S1"
    # xfm = "fullhead"
    # roi = "TEST"

    # def zoom_to_roi(subject, roi, hem, margin=10.0):
    #     roi_verts = cortex.get_roi_verts(subject, roi)[roi]
    #     roi_map = cortex.Vertex.empty(subject)
    #     roi_map.data[roi_verts] = 1

    #     (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat",
    #                                                                 nudge=True)
    #     sel_pts = dict(left=lflatpts, right=rflatpts)[hem]
    #     roi_pts = sel_pts[np.nonzero(getattr(roi_map, hem))[0],:2]

    #     xmin, ymin = roi_pts.min(0) - margin
    #     xmax, ymax = roi_pts.max(0) + margin
    #     plt.axis([xmin, xmax, ymin, ymax])

    # Create dataset
    # data = cortex.Volume.random('S1', 'fullhead')

    # Plot it using quickflat
    # cortex.quickshow(corrvol)

    # Zoom on just one region
    # zoom_to_roi('S1', 'TEST', 'left')

    # plt.show()

    # Get the map of which voxels are inside of our ROI
    # roi_masks = cortex.utils.get_roi_masks(subject, xfm,
    #                                     roi_list=[roi],
    #                                     gm_sampler='cortical-conservative', # Select only voxels mostly within cortex
    #                                     split_lr=False, # No separate left/right ROIs
    #                                     threshold=None, # Leave roi mask values as probabilities / fractions
    #                                     return_dict=True
    #                                     )

    # # Plot the mask for one ROI onto a flatmap
    # roi_data = cortex.Volume(roi_masks[roi], subject, xfm,
    #                         vmin=0, # This is a probability mask, so only
    #                         vmax=1, # so scale btw zero and one
    #                         cmap="inferno", # For pretty
    #                         )

    # cortex.quickflat.make_figure(roi_data,
    #                             thick=1, # select a single depth (btw white matter & pia)
    #                             sampler='nearest', # no interpolation
    #                             with_curvature=True,
    #                             with_colorbar=True,
    #                             )

    # plt.show()
    
    
    # Plot correlation flatmap
    # cortex.quickshow(corrvol, with_rois=True, with_labels=True)
    # plt.show()
    # cortex.quickshow(corrvol, with_rois=True, with_labels=True)
    # plt.show()
    # Show the data on the 3D inflated cortical surface
    # cortex.webshow(data=corrvol, open_browser=False, recache = True)
    # HTML("<a target='_blank' href='https://127.0.0.1/:8000'>Click here for viewer</a>")
    # print("Server is running. Press Ctrl+C to stop.")
    # time.sleep(3600)
    log_file = f'./{config["corpus_type"]}/sub-{args.sub}/log.txt'
    with open(log_file, 'a') as f:
        f.write(f'Model: {config["model"]}\n')
        f.write(f'Corpus Type: {config["corpus_type"]}\n')
        f.write(f'Model Type: {config["model_type"]}\n')
        f.write(f'Token Type: {config["token_type"]}\n')
        f.write(f'Vocabulary: {config["vocab"]}\n')
        f.write(f'fMRI file: "alignment/data/sub-{args.sub}/sub-{args.sub}_{args.roi}.hf5\n')
        f.write(f'Train Linear Model Correlations for subject {args.sub} and ROI {args.roi}:\n')
        f.write(f'zRresp shape: {zRresp.shape}\n')
        f.write(f'zPresp shape: {zPresp.shape}\n')
        f.write(f'mask shape: {mask.shape}\n')
        f.write(f'Corr file: {config["corpus_type"]}/sub-{args.sub}/corrs_{args.sub}_{args.roi}_{config["model_type"]}.npy\n')
        f.write(f'Weight file: {config["corpus_type"]}/sub-{args.sub}/wt_{args.sub}_{args.roi}_{config["model_type"]}.pkl\n')




if __name__ == '__main__':

    main()