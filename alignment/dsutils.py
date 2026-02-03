import numpy as np
import itertools as itools
from DataSequence import DataSequence
import torch
import random

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

def make_word_ds(vocab_file, flag, grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        print(f"Story {st}: {len(grtranscript)} words, {len(goodtranscript)} after filtering.")
        
        d = DataSequence.from_grid(vocab_file, flag, goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneme_ds(grids, trfiles):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        d = DataSequence.from_grid(grtranscript, trfiles[st][0])
        ds[st] = d

    return ds

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D',
    'DH', 'EH', 'ER',   'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH',
    'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as e:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phonemes2(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.upper().strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def make_semantic_model(ds, model, device):
    newdata = []
    l, r = 0, 1
    seq_len = 20
    # random.shuffle(ds.data)
    # print(ds.data)
    while r<=len(ds.data):
        batch_size = 1
        hidden = model.init_hidden(batch_size, device)
        seq = ds.data[l:r]
        if r >= seq_len:
            l += 1
            r += 1
        else:
            r += 1
        # random.shuffle(seq)
        with torch.no_grad():
            indices = ds.encode(" ".join(seq))
            src = torch.LongTensor([indices]).to(device)
            _, _, hidden = model(src, hidden)

        # print(hidden)
        v = hidden[-1][:, -1, :].detach().cpu().view(-1).numpy()
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, data_times=ds.data_times, tr_times=ds.tr_times)

def make_semantic_model_word(ds, model, device):
    newdata = []
    l, r = 0, 1
    seq_len = 20
    while r<=len(ds.data):
        batch_size = 1
        hidden = model.init_hidden(batch_size, device)
        seq = ds.data[l:r]
        if r >= seq_len:
            l += 1
            r += 1
        else:
            r += 1
        with torch.no_grad():
          indices = ds.tokenize(" ".join(seq))
          src = torch.LongTensor([indices]).to(device)
          _, _, hidden = model(src, hidden)
        v = hidden[-1][:, -1, :].detach().cpu().view(-1).numpy()
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, data_times=ds.data_times, tr_times=ds.tr_times)

def make_semantic_model_gpt(ds, model, device):
    newdata = []
    l, r = 0, 1
    seq_len = 20
    # random.shuffle(ds.data)
    # print(ds.data)
    while r<=len(ds.data):
        seq = ds.data[l:r]
        if r >= seq_len:
            l += 1
            r += 1
        else:
            r += 1
        # random.shuffle(seq)
        with torch.no_grad():
            indices = ds.encode(" ".join(seq))
            src = torch.LongTensor([indices]).to(device)
            last_hidden_state, logits, loss = model(src)
        v = last_hidden_state[:, -1, :].detach().cpu().view(-1).numpy()
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, data_times=ds.data_times, tr_times=ds.tr_times)

def make_semantic_model_gpt_word(ds, model, tokenizer, device):
    newdata = []
    l, r = 0, 1
    seq_len = 20
    while r<=len(ds.data):
        seq = ds.data[l:r]
        if r >= seq_len:
            l += 1
            r += 1
        else:
            r += 1
        random.shuffle(seq)
        with torch.no_grad():
            inputs = ds.gpt_tokenize(" ".join(seq), tokenizer).to(device)
            # src = torch.LongTensor([indices]).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            
        v = last_hidden_state[:, -1, :].detach().cpu().view(-1).numpy()
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, data_times=ds.data_times, tr_times=ds.tr_times)


def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])
