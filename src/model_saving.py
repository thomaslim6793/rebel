from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import omegaconf

# Purpose of this module: Transform the pre-trained BART model into 
# the REBEL model by reconfiguring the tokenizer.
# So this module is meant to be run only once, and is not for periodically
# saving the model during training, despite the name of the file.

# Normally we would need to do the following:
# 1. Change the tokenizer to include the special tokens we need for our task.
# 2. Change the model to include the new special tokens in its embedding layer,
# to reflect the new tokens we added in the tokenizer.
# 3. Change the head layer with the our own downstream task, if the task of this
# base model is different from our downstream task.

# In our case, we are using the seq2seq variant of BART, and our own task 
# is seq2seq, so we don't need to do any work to change the head layer. 

# So then this module does the following:
# 1. Load the seq2seq BART model and its tokenizer.
# 2. Change the tokenizer to include the special tokens we need for our task.
# 3. Change the model to include the new special tokens in its embedding layer,
# 4. Save this model and tokenizer as our not-yet-pre-trained REBEL model.

# Note that I say the saved REBEL module/model is not pre-trained because the weights
# are that of the pre-trained BART model. To get the 'pre-trained REBEL model' we would 
# need to train this model on the REBEL dataset. 

# 1. Load the config of the BART model, and define or override some default parameters 
# so that the config corresponds to our design.
config = AutoConfig.from_pretrained(
    'facebook/bart-large',
    decoder_start_token_id = 0,
    early_stopping = False,
    no_repeat_ngram_size = 0,
)
# 2. Load the tokenizer from BART model and add the special tokens
tokenizer = AutoTokenizer.from_pretrained(
    'facebook/bart-large',
    use_fast=True,
    additional_special_tokens = ['<obj>', '<subj>', '<triplet>']
)

# 3. Load the BART model: architecture + weights
model = AutoModelForSeq2SeqLM.from_pretrained(
    'facebook/bart-large',
    config=config,
)
# 4. Resize the embeddings layer to include the three new tokens to have dim=[tokenizer.vocab_size + 3, embedding vector dim]
model.resize_token_embeddings(len(tokenizer))

# 5. Load the configurations required to pre-train the not-yet-pre-trained REBEL model to pre-trained REBEL model.
# Here conf is loaded from a YAML file using OmegaConf.load('outputs/XXXX-XX-XX/XX-XX-XX/.hydra/config.yaml'). 
# This is typically broader and includes configurations for the training environment, data handling, experiment management, 
# and possibly hyperparameters that are not specific to the model's architecture but to how the model is trained and evaluated.
conf = omegaconf.OmegaConf.load('outputs/XXXX-XX-XX/XX-XX-XX/.hydra/config.yaml')
# 6. Create a base module with the configurations, tokenizer, and model.
pl_module = BasePLModule(conf, config, tokenizer, model)
# 7. Load the model from a checkpoint and supplying it with the configurations, tokenizer, and the model architecture.
# The pl_module instance is already initialized with the pre-trained BART model, so this may seem redundant, since this line
# is reassigning the pl_module variable with a new instance from a checkpoint, but it is an option if we already have 
# a checkpoint for BART with an even better weights initialization.
# Also note that checkpoint by itself doesn't contain the model architecture, when loading a model from a checkpoint, we 
# need to supply the model architecture as well. This is what the `model = model` keyword argument is doing.
model = pl_module.load_from_checkpoint(checkpoint_path = 'outputs/XXXX-XX-XX/XX-XX-XX/experiments/dataset/last.ckpt', config = config, tokenizer = tokenizer, model = model)

# 8. Save the model and tokenizer. Now we have a REBEL model that is not pre-trained, but is ready to be pre-trained.
model.model.save_pretrained('../model/MODEL-NAME')
model.tokenizer.save_pretrained('../model/MODEL-NAME')