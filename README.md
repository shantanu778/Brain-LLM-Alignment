# Brain-LLM Alignment: Mapping Syntactically Corrupted Language Model Representations to Human Brain Responses

## 1. Project Overview

This project investigates the alignment between language model representations and human brain responses (fMRI) when syntactic linguistic information is systematically corrupted. The core research question is: **how does selectively removing specific syntactic categories (nouns, verbs, adjectives, etc.) from language model training data affect the alignment between model representations and neural activity?**

### Approach

1. **Text corruption**: Replace words of a target POS category (e.g., all nouns) with placeholder tokens (e.g., `NNOUN`) in a ~12M-word training corpus drawn from Wikipedia, Project Gutenberg, and UKWaC.
2. **Model training**: Train LSTM and GPT-2 models from scratch on each corrupted corpus for next-word prediction. A pre-trained Gemma-3 4B model is also fine-tuned via QLoRA on a subset of conditions.
3. **Representation extraction**: Extract hidden-state representations from each model on the *Moth Radio Hour* naturalistic listening stories.
4. **Encoding model**: Fit a ridge regression (bootstrap ridge with SVD) mapping model representations to voxel-level fMRI responses across 32 SENSAAS brain ROIs for 9 subjects.
5. **Analysis**: Compare encoding accuracy (Pearson correlation) across corruption conditions, models, and brain regions.

### Corruption Conditions

| Condition | POS Removed | Placeholder |
|-----------|------------|-------------|
| `plain` | None (baseline) | — |
| `nounless` | Nouns | `NNOUN` |
| `verbless` | Verbs | `NVERB` |
| `adjless` | Adjectives | `NADJ` |
| `advless` | Adverbs | `NADV` |
| `pronounless` | Pronouns | `NPRON` |
| `detless` | Determiners | `NDET` |
| `contless` | Content words | `NCONT` |
| `functless` | Function words | `NFUNCT` |

Each condition also has a **random-replacement baseline** (`*-equiv`) where the same number of words are replaced with random vocabulary tokens instead of fixed placeholders.

### Models

- **LSTM**: 2-layer, 768-dim hidden, character-level tokenization
- **GPT-2**: 12-layer, 12-head, 768-dim, character-level tokenization (nanoGPT architecture)
- **Gemma**: google/gemma-3-4b-pt, QLoRA fine-tuned (4-bit NF4), BPE tokenization

---

## 2. Study Methodology

### fMRI Data

The Moth Radio Hour dataset ([OpenNeuro ds003020 v4.0.0](https://openneuro.org/datasets/ds003020/versions/4.0.0)) provides fMRI recordings from 9 subjects (UTS01–UTS09) listening to 11 naturalistic stories (10 training stories + 1 test story "wheretheressmoke").

### Brain ROIs

32 ROIs from the SENSAAS atlas (Labache et al. 2019), covering temporal, frontal, parietal, insular, and subcortical language-related regions.

### Representation Extraction

- Context window: 20 words
- For LSTM: final hidden state at each word position
- For GPT-2: hidden state from a configurable layer (default: layer 5 or 6)
- For Gemma: last-token pooling from the final hidden layer
- Downsampling: Lanczos interpolation (3-lobe) to TR resolution (~2s)
- FIR delays: 4 delays (1–4 TRs) to model hemodynamic lag

### Encoding Model

- Bootstrap ridge regression (5 bootstraps, chunk length 40, 20 chunks)
- Alphas: 10 log-spaced values from 10 to 1000
- Training: 10 stories (~3,700 TRs); Test: 1 story ("wheretheressmoke")
- Evaluation metric: Pearson correlation per voxel on the held-out test story
- Random seeds: 42 (numpy, random module) for reproducibility

### Statistical Testing

- FSL `randomise` for non-parametric permutation testing (5,000 permutations, TFCE)
- **[INCOMPLETE]**: Statistical testing scripts are not yet included in the repository.

---

## 3. Repository Structure

```
Brain-LLM-Alignment/
├── README.md                       # This file
├── requirements.txt                # Python dependencies (pinned versions)
├── run.sh                          # Orchestration: loop subjects × ROIs for corruption conditions
├── run_ran.sh                      # Orchestration: same loop for random-replacement baselines
├── corr.py                         # Aggregate per-subject/ROI .npy correlations into .xlsx
│
├── alignment/                      # Core encoding pipeline
│   ├── main.py                     # Entry point: load config → extract reps → ridge regression → save correlations
│   ├── lm_rep.py                   # LLM representation extraction (LSTM/GPT-2/Gemma) → HDF5
│   ├── ridge.py                    # Bootstrap ridge regression with SVD
│   ├── dsutils.py                  # Semantic model construction (word→embedding sequences)
│   ├── DataSequence.py             # Time-aligned data sequences with tokenization methods
│   ├── npp.py                      # z-scoring and preprocessing utilities
│   ├── interpdata.py               # Lanczos interpolation for downsampling
│   ├── stimulus_utils.py           # Load TextGrids and TRfiles
│   ├── textgrid.py                 # TextGrid parser
│   ├── util.py                     # FIR delay matrix construction (make_delayed)
│   ├── utils.py                    # Additional utilities
│   └── data/                       # Preprocessed fMRI data (HDF5, per subject × ROI)
│       ├── grids/                  # Praat TextGrid files (word-level timing per story)
│       ├── trfiles/                # TR timing files per story
│       ├── sub-UTS01/ ... sub-UTS09/  # Per-subject ROI HDF5 files
│       └── respdict.json           # Response dictionary metadata
│
├── lstm/                           # LSTM model definition and training
│   ├── main.py                     # Training loop with early stopping
│   ├── models.py                   # Custom_LSTM class (2-layer, 768-dim)
│   ├── data_process.py             # Corpus preprocessing
│   ├── inference.py                # Inference utilities
│   └── utils.py                    # Data loading, batching, vocab creation
│
├── gpt2/                           # GPT-2 model definition and training
│   ├── main.py                     # GPTConfig, GPT class, training loop (nanoGPT-style)
│   ├── prepare.py                  # Data preparation
│   └── sample.py                   # Text generation / sampling
│
├── corruption-dirs/                # POS-corrupted experiment data and results
│   ├── plain/                      # Uncorrupted baseline
│   ├── nounless/                   # Nouns removed
│   ├── verbless/                   # Verbs removed
│   ├── adjless/                    # Adjectives removed
│   ├── advless/                    # Adverbs removed
│   ├── pronounless/                # Pronouns removed
│   ├── detless/                    # Determiners removed
│   ├── contless/                   # Content words removed
│   └── functless/                  # Function words removed
│   Each contains:
│     ├── {guthenberg,wiki,ukwac}_{condition}.txt  # Corrupted training corpora
│     ├── {train,val,test}.npy                     # Tokenized splits
│     ├── {lstm,gpt2}/char/models/                 # Trained checkpoints + vocab + config
│     ├── gemma/bpe/models/                        # Gemma fine-tuned adapter (some conditions)
│     ├── sub-UTS01/ ... sub-UTS09/                # Per-subject correlation outputs
│     └── {model}_correlations.xlsx                # Aggregated results
│
├── Random-corruption-dirs/         # Random-replacement baselines
│   ├── replace-words-3.py          # Script to create random replacements
│   ├── make-random-replacements.sh # Shell wrapper for batch processing
│   └── {condition}-equiv/          # Mirror structure of corruption-dirs/
│
├── LeBelAetal2023_moth/            # Preprocessed fMRI data from LeBel et al. 2023
│   └── (subject data, TextGrids, stimulus files)
│
├── ds003020/                       # Raw fMRI dataset (via datalad, not in git)
│
├── gemma3-finetuned/               # Gemma fine-tuning outputs
│
└── moth_fMRI_by_ROI_method_subUTS01.ipynb  # Jupyter notebook: ROI-based fMRI processing pipeline
```

---

## 4. Requirements and Environment Setup

### Hardware

- GPU with CUDA support required for model training and Gemma inference
- Server used: 2× NVIDIA RTX 3090 (CIMEC clicgpu4)
- Disk: ~50GB for data, checkpoints, and results

### Software Prerequisites

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9.21 | Primary runtime |
| PyTorch | ≥2.1.1 (CUDA 12.1) | Model training/inference |
| FSL | 6.x | Non-parametric permutation testing (`randomise`, `fslmaths`) |
| datalad + git-annex | Latest | fMRI data download |
| MATLAB + SPM25 + CAT12 | Optional | ROI registration to MNI space |

### Environment Setup

```bash
# Create conda environment
conda create -n brain_llm_3.9 python=3.9.21
conda activate brain_llm_3.9

# Install PyTorch with CUDA 12.1
pip install torch==2.1.1+cu121 torchaudio==2.1.1+cu121 torchvision==0.16.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note**: `pycortex` requires additional configuration for cortical surface visualization. FSL must be installed separately (not pip-installable).

---

## 5. Data Acquisition and Preparation

### 5.1 fMRI Data (Moth Radio Hour)

The raw fMRI data is publicly available on OpenNeuro:

```bash
# Install datalad (see https://handbook.datalad.org/en/latest/intro/installation.html)
pip install datalad

# Clone the dataset
datalad install https://github.com/OpenNeuroDatasets/ds003020.git
cd ds003020
datalad get .  # Downloads all files (~large)
```

Reference: [OpenNeuro ds003020 v4.0.0](https://openneuro.org/datasets/ds003020/versions/4.0.0)
Documentation: [datalad handbook — OpenNeuro use case](https://handbook.datalad.org/en/latest/usecases/openneuro.html)

### 5.2 Preprocessed fMRI Data

The repository expects preprocessed per-subject, per-ROI HDF5 files at:

```
alignment/data/sub-{UTSXX}/sub-{UTSXX}_{ROI_NAME}.hf5
```

Each HDF5 file contains:
- `zRresp`: z-scored training responses (n_training_TRs × n_voxels)
- `zPresp`: z-scored test responses (n_test_TRs × n_voxels)
- `mask`: binary voxel mask

These files are derived from the LeBel et al. 2023 preprocessing pipeline. The directory `LeBelAetal2023_moth/` contains intermediate data from that pipeline.

> **[UNRESOLVED]**: The exact preprocessing steps from raw ds003020 NIfTI files to the per-ROI HDF5 files are not fully documented. The pipeline involves:
> 1. Standard fMRI preprocessing (motion correction, spatial normalization)
> 2. ROI extraction using the 32 SENSAAS atlas parcels
> 3. z-scoring and train/test splitting by story
> 4. The Jupyter notebook `moth_fMRI_by_ROI_method_subUTS01.ipynb` documents parts of this process

### 5.3 Training Corpora

Each corruption condition directory contains pre-corrupted text files:

```
corruption-dirs/{condition}/{guthenberg,wiki,ukwac}_{condition}.txt
```

The original corpora (~12M words total) are drawn from:
- **Wikipedia** dump
- **Project Gutenberg** texts
- **UKWaC** web-crawled corpus

POS tagging and replacement are performed before tokenization. The random-replacement baseline corpora are generated by `Random-corruption-dirs/replace-words-3.py`.

### 5.4 TextGrids and TR Files

Word-level timing annotations (Praat TextGrids) and TR timing files are stored in:
```
alignment/data/grids/    # One .TextGrid per story
alignment/data/trfiles/  # One .trfile per story
```

These are provided with the repository and are required for aligning model representations to fMRI time points.

### 5.5 Expected Directory Layout

Place the following before running the pipeline:

```
Brain-LLM-Alignment/
├── alignment/data/sub-UTS01/ ... sub-UTS09/   # Preprocessed fMRI (HDF5)
├── alignment/data/grids/                       # TextGrid files
├── alignment/data/trfiles/                     # TR timing files
├── corruption-dirs/{condition}/{train,val,test}.npy  # Tokenized training data
└── corruption-dirs/{condition}/{corpus}_{condition}.txt  # Raw corrupted text
```

---

## 6. Step-by-Step Reproduction Instructions

All commands assume the working directory is the repository root: `Brain-LLM-Alignment/`.

### Step 1: Train a Language Model

Choose a corruption condition and model type. Each condition has a config JSON file.

**Train LSTM** (example: plain baseline):
```bash
python lstm/main.py -f corruption-dirs/plain/config_lstm.json
```

**Train GPT-2** (example: plain baseline):
```bash
python gpt2/main.py -f corruption-dirs/plain/config_gpt2.json
```

- **Input**: `corruption-dirs/{condition}/{train,val,test}.npy`, vocabulary pickle
- **Output**: `corruption-dirs/{condition}/{model}/char/models/best-val-{model}-{seq_len}.pt`
- **Config parameters**: `epochs`, `batch_size`, `seq_len`, `hidden_layer`, `lr`, etc.
- **Success check**: Checkpoint `.pt` file appears in `models/` directory; training loss decreases

**Fine-tune Gemma** (example: functless):
```bash
python gpt2/main.py -f corruption-dirs/functless/config_gemma.json
```
> **Note**: Gemma fine-tuning requires ~24GB VRAM (QLoRA 4-bit). The Gemma base model (`google/gemma-3-4b-pt`) is downloaded from Hugging Face on first run.

### Step 2: Run the Encoding Pipeline (All Subjects × All ROIs)

**For corruption conditions:**
```bash
bash run.sh corruption-dirs/{condition} {model_type} char
# Example:
bash run.sh corruption-dirs/plain gpt2 char
```

**For random-replacement baselines:**
```bash
bash run_ran.sh Random-corruption-dirs/{condition}-equiv {model_type}
# Example:
bash run_ran.sh Random-corruption-dirs/nounless-equiv gpt2
```

These scripts loop over 9 subjects × 32 ROIs, calling `alignment/main.py` for each combination.

- **Input**: Trained model checkpoint, config JSON, preprocessed fMRI HDF5 files
- **Output per run**:
  - `{condition}/{model}/char/models/stories_rep_layer_{N}.hf5` (LM representations, created once)
  - `{condition}/sub-{UTSXX}/corrs_{UTSXX}_{ROI}_{model}_layer_{N}.npy` (per-voxel correlations)
  - `{condition}/sub-{UTSXX}/wt_{UTSXX}_{ROI}_{model}_layer_{N}.pkl` (ridge regression weights)
- **Success check**: `.npy` correlation files exist for all 9×32 = 288 subject-ROI combinations
- **Runtime**: ~2–4 hours per condition per model on 2× RTX 3090

### Step 3: Aggregate Results

```bash
python corr.py -m {model_type} -c corruption-dirs/{condition} -l {layer}
# Example:
python corr.py -m gpt2 -c corruption-dirs/plain -l 6
```

- **Input**: All 288 `.npy` files from Step 2
- **Output**: `corruption-dirs/{condition}/{model}_layer_{layer}_correlations.xlsx`
- **Success check**: Excel file with mean correlations per subject × ROI

### Step 4: Run for a Single Subject/ROI (alternative to Step 2)

```bash
python alignment/main.py \
  -f corruption-dirs/{condition}/{model}/char/models/config.json \
  --sub UTS01 \
  --roi G_Temporal_Sup-4.nii
```

---

## 7. Reproducing the Main Results

### Available Results

The repository contains pre-computed correlation spreadsheets for completed conditions:

| Condition | LSTM | GPT-2 | Gemma |
|-----------|------|-------|-------|
| plain | ✅ | ✅ | ✅ |
| nounless | ✅ | ✅ | ✅ |
| verbless | ✅ | ✅ | ✅ |
| adjless | ✅ | ✅ | — |
| advless | ✅ | ✅ | — |
| pronounless | ✅ | ✅ | — |
| detless | ✅ | ✅ | — |
| contless | ✅ | ✅ | ✅ |
| functless | ✅ | ✅ | ✅ |

Random-replacement baselines (`*-equiv`) are complete for LSTM and GPT-2 across all conditions.

### Key Output Files

- **Per-condition correlation spreadsheet**: `corruption-dirs/{condition}/{model}_correlations.xlsx`
  - Columns: `corpus_type`, `model_type`, `roi`, `subject`, `mean_corrs`, `std`
- **Per-voxel correlations**: `corruption-dirs/{condition}/sub-{UTSXX}/corrs_{UTSXX}_{ROI}_{model}_layer_{N}.npy`
- **Ridge weights**: `corruption-dirs/{condition}/sub-{UTSXX}/wt_{UTSXX}_{ROI}_{model}_layer_{N}.pkl`

### Comparing Results

To compare reproduced results with existing ones, load the pre-computed `.xlsx` files and compare mean correlations per ROI and subject. Due to the random seed fix (seed 42), new runs should produce near-identical results for the bootstrap ridge regression. Minor floating-point differences may occur across hardware.

---

## 8. Configuration and Customization

### Config JSON Structure

Each condition/model combination uses a JSON config file:

```json
{
    "vocab": "vocab",
    "epochs": 20,
    "corpus_type": "corruption-dirs/plain",
    "model_type": "gpt2",
    "model": "best-val-gpt-256",
    "token_type": "char",
    "batch_size": 16,
    "hidden_layer": 5,
    "seq_len": 256
}
```

| Parameter | Description | Required |
|-----------|-------------|----------|
| `corpus_type` | Path to condition directory | Yes |
| `model_type` | `lstm`, `gpt2`, or `gemma` | Yes |
| `model` | Checkpoint name (without `.pt`) | Yes |
| `token_type` | `char` (LSTM/GPT-2) or `bpe` (Gemma) | Yes |
| `hidden_layer` | Which layer to extract representations from | Yes |
| `vocab` | Vocabulary pickle name | Yes |
| `epochs` | Training epochs | Training only |
| `batch_size` | Batch size | Training only |
| `seq_len` | Sequence length for training | Training only |
| `fine_tuned` | `true`/`false` for Gemma pre-trained vs fine-tuned | Gemma only |

### LSTM-Specific Parameters (in config)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 768 | Embedding dimension |
| `hidden_dim` | 768 | Hidden state dimension |
| `num_layers` | 2 | Number of LSTM layers |
| `dropout_rate` | 0.4 | Dropout rate |
| `tie_weights` | true | Tie embedding and output weights |

### Ridge Regression Parameters (hardcoded in `alignment/main.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `alphas` | `logspace(1, 3, 10)` | Regularization strengths |
| `nboots` | 5 | Bootstrap cross-validation folds |
| `chunklen` | 40 | Chunk length for bootstrap sampling |
| `nchunks` | 20 | Number of chunks per bootstrap |
| `singcutoff` | 1e-10 | SVD singular value cutoff |
| `random seed` | 42 | Reproducibility seed |

### FIR Delays (hardcoded in `alignment/lm_rep.py`)

| Parameter | Value |
|-----------|-------|
| `ndelays` | 4 |
| `delays` | 1, 2, 3, 4 TRs |
| `trim` | 5 TRs |

---

## 9. Reproducibility, Limitations, and Current Status

### Reproducibility

- **Random seeds**: Fixed at 42 for ridge regression (`ridge.py`, `dsutils.py`) and 0 for model inference (`lm_rep.py`).
- **Training seeds**: `torch.manual_seed(0)` in LSTM training; GPT-2 training uses the same.
- **Hardware dependence**: Floating-point results may differ slightly across GPU architectures. Gemma 4-bit quantization output depends on `bitsandbytes` version and GPU.

### Known Limitations

1. **fMRI preprocessing not fully documented**: The pipeline from raw NIfTI (ds003020) to per-ROI HDF5 files requires MATLAB/SPM/CAT12 and is only partially documented in the Jupyter notebook.
2. **Gemma results incomplete**: Only 4 of 9 corruption conditions have Gemma results (plain, nounless, verbless, contless, functless).
3. **Statistical testing scripts missing**: FSL `randomise` commands for group-level permutation testing are not included.
4. **Wilcoxon signed-rank test script missing**: No script for pairwise statistical comparison between conditions.
5. **No end-to-end workflow script**: Individual steps must be run manually in sequence.
6. **Training corpora not redistributable**: The raw UKWaC corpus requires a separate license. Wikipedia and Gutenberg texts are public domain, but the corrupted versions are generated by scripts not fully included.

### Current Project Status

- **Complete**: Model training (LSTM, GPT-2) for all 9 conditions + random baselines; encoding pipeline for LSTM/GPT-2 across all conditions
- **Partial**: Gemma fine-tuning and encoding (4/9 conditions); layer-specific GPT-2 analysis
- **Not started**: Statistical testing pipeline, Wilcoxon comparison scripts, publication figures

### Unverified Instructions

The following instructions in this README have **not** been tested end-to-end from a clean checkout:
- Conda environment creation and full dependency installation
- datalad dataset download
- Full training from raw corpora to final checkpoints
- Gemma fine-tuning pipeline

The encoding pipeline (`run.sh` → `alignment/main.py` → `corr.py`) has been verified to work on the CIMEC server with existing checkpoints and preprocessed data.

---

## 10. Citation and Licensing

### Dataset Citation

> LeBel, A., et al. (2023). A natural language fMRI dataset for voxelwise encoding models. *Scientific Data*, 10, 555. [OpenNeuro ds003020 v4.0.0](https://openneuro.org/datasets/ds003020/versions/4.0.0)

### Atlas Citation

> Labache, L., et al. (2019). A SENtence Supramodal Areas AtlaS (SENSAAS) based on multiple task-induced activation mapping and graph analysis of intrinsic connectivity in 144 healthy right-handers. *Brain Structure and Function*, 224, 859–882.

### Software

- nanoGPT architecture adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Encoding model approach based on the Huth Lab framework

### Project Citation

> **[TODO]**: Add citation when the manuscript is published.

### License

> **[TODO]**: No license file is present in the repository. Add a LICENSE file before public release.
