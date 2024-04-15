import omegaconf
import hydra
import torch
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pl_data_modules import BasePLDataModule
from pl_modules import BasePLModule
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from generate_samples import GenerateTextSamplesCallback

# Evaluate the model on unseen data using predefined metrics. 
def test(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.seed)

    config = AutoConfig.from_pretrained(
        conf.config_name if conf.config_name else conf.model_name_or_path,
        decoder_start_token_id = 0,
        early_stopping = False,
        no_repeat_ngram_size = 0,
        # cache_dir=conf.cache_dir,
        # revision=conf.model_revision,
        # use_auth_token=True if conf.use_auth_token else None,
    )
    
    tokenizer_kwargs = {
        "use_fast": conf.use_fast_tokenizer,
        "additional_special_tokens": ['<obj>', '<subj>', '<triplet>'],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        conf.tokenizer_name if conf.tokenizer_name else conf.model_name_or_path,
        **tokenizer_kwargs
    )
    if conf.dataset_name.split('/')[-1] == 'conll04_typed.py':
        tokenizer.add_tokens(['<peop>', '<org>', '<other>', '<loc>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'nyt_typed.py':
        tokenizer.add_tokens(['<loc>', '<org>', '<per>'], special_tokens = True)
    if conf.dataset_name.split('/')[-1] == 'docred_typed.py':
        tokenizer.add_tokens(['<loc>', '<misc>', '<per>', '<num>', '<time>', '<org>'], special_tokens = True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        conf.model_name_or_path,
        config=config,
    )
    # Resize model embedding to include new tokens
    model.resize_token_embeddings(len(tokenizer))

    # data module declaration
    pl_data_module = BasePLDataModule(conf, tokenizer, model)

    # main module declaration
    if conf.checkpoint_path and conf.finetune:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_abs_path = os.path.join(base_path, conf.checkpoint_path)
        checkpoint = torch.load(checkpoint_abs_path, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict']
        # Adjust the keys by removing the first occurrence of 'model.' The issue I had is that the state dict keys of
        # the checkpoint was prefixed with `model.`, so I had to remove it to match the model's state dict keys.
        # E.g. A key in the model's state dict is `model.shared.weight`, but in the checkpoint it is `model.model.shared.weight`.
        adjusted_state_dict = {key.partition('model.')[2]: value for key, value in state_dict.items() if key.startswith('model.')}
        model.load_state_dict(adjusted_state_dict)
        pl_module = BasePLModule(conf, config, tokenizer, model)
    else:
        pl_module = BasePLModule(conf, config, tokenizer, model)  # Adjust as per your constructor requirements

    pl_module.hparams.test_file = pl_data_module.conf.test_file
    # trainer

    # Check if any GPUs are available
    gpu_count = torch.cuda.device_count()
    accelerator = 'gpu' if gpu_count > 0 else 'cpu'

    # Configure the trainer to use GPU if available, otherwise CPU
    trainer = pl.Trainer(
        devices=conf.gpus if gpu_count > 0 else 1,  # Use all GPUs available or 1 CPU
        accelerator=accelerator
    )

    # Manually run prep methods on DataModule
    pl_data_module.prepare_data()
    pl_data_module.setup(stage='test')

    #The pl.Trainer object in PyTorch Lightning is designed not just for training but 
    # also to facilitate validation and testing. It provides a standardized way to run 
    # these processes, ensuring that they leverage the same distributed and accelerated 
    # environment setup that is used during training.
    trainer.test(pl_module, dataloaders=pl_data_module.test_dataloader())

@hydra.main(config_path='../conf', config_name='root')
def main(conf: omegaconf.DictConfig):
    test(conf)


if __name__ == '__main__':
    main()
