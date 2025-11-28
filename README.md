# Brain-LLM-Alignment
In this study, we investigate the alignment between the representation of language models and human brains while syntactic lingustic information are corrupted. To train language models, we collect a diverse corpus of approximately 12 million words from sources like Wikipedia, the Gutenberg Project, and the Ukwac dataset and replace the syntactic information such as Noun, Pronoun, Adjective and so on with specific words for example NNOUN, NPRON, NADJ and so on respectively. We then train LSTM and GPT-2 models from scratch on our distorted text for next word prediction. For brain responses, we leverage The Moth Radio Hour dataset, a radio program where storytellers tell true, autobiographical stories in front of a live audience. To map language model representation with brain responses, we first extract the representation of our language models on radio dataset with a context length of 20 words. Then, an encoding model has been used to map language representation with brain responses. We observe a decent alignment among different regions of the brain with language models. 

# Installation 
```
pip install -r requirements.txt
```

# Usage
Before training a new model change the config file accordingly. Such as, courpus_type: "contless", and "model_type": "gpt2",

# Train
## For GPT2

Step 1. Place your text file in dedicated folder and prepare your dataset using following command:

```
python gpt2/prepare.py  -f gpt2/config.json 

```
Step 2. Train Language models, 

```
python gpt2/main.py  -f gpt2/config.json 

```

## For LSTM

```
python lstm/main.py  -f lstm/config.json 

```
