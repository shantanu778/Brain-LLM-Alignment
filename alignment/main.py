import os
import sys
import __main__
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
import numpy as np
import pickle
import json
import tables
from ridge import bootstrap_ridge
from lm_rep import llm_representation
from lstm.models import Custom_LSTM
from gpt2.main import GPTConfig, GPT

__main__.GPTConfig = GPTConfig
__main__.GPT = GPT

def main():
    parser = argparse.ArgumentParser(description="Train linear model to predict fMRI responses from LLM representations")
    parser.add_argument('-f', '--config_file', type=str, required=True,
                        help="Path to configuration JSON file")
    parser.add_argument('--sub', type=str, required=True, help="Subject ID")
    parser.add_argument('--roi', type=str, required=True, help="Region of Interest")
    parser.add_argument('--test', default=False, action='store_true', help="Flag to do Test")
    parser.add_argument('--plain', default=False, action='store_true', help="Flag to Use Linear Model trained on plain text")
    args = parser.parse_args()
    f = open(args.config_file)
    config = json.load(f)
    print(config)

    lm_rep_file = Path(f'{config["corpus_type"]}/{config["model_type"]}/{config["token_type"]}/models/stories_rep.hf5')
    if not lm_rep_file.exists():
        # Pass your arguments as a list of strings, exactly as you would type them in the terminal
        lm_args = ["--file_name", f"{config['corpus_type']}/{config['model_type']}/{config['token_type']}/models/config.json", '--sub', f"{args.sub}"]
        llm_representation(lm_args)

    delstim = tables.open_file(lm_rep_file)
    delRstim = delstim.root.delRstim.read()
    delPstim = delstim.root.delPstim.read()
    print("delRstim shape:", delRstim.shape)
    print("delPstim shape:", delPstim.shape)    

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