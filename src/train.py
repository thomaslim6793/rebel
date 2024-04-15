import omegaconf
import hydra
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

def train(conf: omegaconf.DictConfig) -> None:
    # Use a fixed seed for reproducibility for pseudo-random number generation
    pl.seed_everything(conf.seed)
    
    # Load the model-level configuration from the checkpoint, specific to HuggingFace framework.
    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        dropout=conf.dropout,
        forced_bos_token_id=None,
    )
    
    # Load the tokenizer-level configuration from the checkpoint, specific to HuggingFace framework. 
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        # Here the tokens for head and tail are legacy and only needed if finetuning over the public REBEL 
        # checkpoint, but are not used. If training from scratch, remove this line and uncomment the next one.
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>', '<head>', '</head>', '<tail>', '</tail>'], 
#         "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    # Load the tokenizer from the checkpoint of the model, specific to HuggingFace framework.
    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )

    # Depending on which dataset is being used, add the special tokens to the tokenizer.
    if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
        tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
        tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
        tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    # Load the model from the checkpoint, specific to HuggingFace framework.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )

    # Resize the token embeddings to match the tokenizer vocabulary size.
    model.resize_token_embeddings(len(tokenizer))

    # Create data module object
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # Create model module object
    pl_module = BasePLModule(conf, config, tokenizer, model)

    # A logger for logging the training process to the Neptune platform.
    wandb_logger = WandbLogger(project = conf.dataset_name.split('/')[-1].replace('.py', ''), name = conf.model_name_or_path.split('/')[-1])

    # Store the callback objects which will be used by pl.Trainer
    # which automatically triggers methods of these callback objects
    # at various points in the training phase. 
    callbacks_store = []

    # Callback object for early stopping.
    if conf.apply_early_stopping:
        callbacks_store.append(
            EarlyStopping(
                monitor=conf.monitor_var,
                mode=conf.monitor_var_mode,
                patience=conf.patience
            )
        )

    # Callback object for saving the model checkpoints at various points in
    # the training timeline.
    callbacks_store.append(
        ModelCheckpoint(
            monitor=conf.monitor_var,
            # monitor=None,
            dirpath=f'experiments/{conf.model_name}',
            save_top_k=conf.save_top_k,
            verbose=True,
            save_last=True,
            mode=conf.monitor_var_mode
        )
    )
    # Callback object for generating text samples at various points in the training timeline.
    # Remember that generating text samples is for diagnostic purpose to see how the model is performing
    # at various points in the training timeline.
    callbacks_store.append(GenerateTextSamplesCallback(conf.samples_interval))
    # Callback object for monitoring the learning rate at various points in the training timeline.
    callbacks_store.append(LearningRateMonitor(logging_interval='step'))

    # Finally we create the Pytorch Lightning Trainer object which will be used to train the model.
    # It takes in various arguments, some which are hardware level (e.g. `gpus`), some which are
    # optimization level (e.g. `accumulate_grad_batches`), some which are for logging (e.g. `logger`),
    # and some which are for checkpointing (e.g. `resume_from_checkpoint`), and finally some are 
    # hyperparameters of training (e.g. `max_steps`).

    # Check if any GPUs are available and CUDA is available.
    if torch.cuda.is_available():
        test_tensor = torch.tensor([1.0]).cuda()
        print("CUDA is available, test tensor:", test_tensor)
    else:
        print("CUDA is not available.")
    gpu_count = torch.cuda.device_count()
    accelerator = 'gpu' if gpu_count > 0 else 'cpu'

    print(f"Number of GPUs detected: {gpu_count}")
    print(f"Selected accelerator: {accelerator}")


    trainer = pl.Trainer(
        devices=conf.gpus if gpu_count > 0 else 1,
        accelerator=accelerator,
        accumulate_grad_batches=conf.gradient_acc_steps,
        gradient_clip_val=conf.gradient_clip_value,
        val_check_interval=conf.val_check_interval,
        callbacks=callbacks_store,
        max_steps=conf.max_steps,
        # max_steps=total_steps,
        precision=conf.precision,
        logger=wandb_logger,
        limit_val_batches=conf.val_percent_check
    )

    # Finally fit this trainer object to the model and datamodule objects. 
    # This will start the training process.
    trainer.fit(pl_module, datamodule=pl_data_module)

# Decorator for using Hydra for configuration management. 
# It transforms this main function it decorates into a Hydra application.
# Simply put, Hydra is just creating the omegaconf.DictConfig object 
# by parsing `../conf` directory and given the `root.yaml` file and 
# then passing this object containing the configuration parameters in
# a hierarchical dictionary format, as argument to `main`. 
@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    train(conf)

if __name__ == '__main__':
    main()
