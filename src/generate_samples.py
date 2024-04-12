from typing import Sequence

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import nn
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import wandb


# Purpose of this module: This module defines a callback class that is used to generate samples and
# log the generated samples during training. That is, this class is used to continuously and periodically 
# generate (or make inference of) the outputs of the training data during training and log the predictions to the Weights and Biases dashboard.
# So this module is a "diagnostic code"; it is not an essential part of the training process, but is used to monitor the training process.

# This is a Callback class of Pytorch Lightning module. This means that it is class that is automatically called by PyTorch Lightning 
# at certain points during training when it is provided as an argument to Pytotorch Lightning's Trainer class. 
# The structure of this class, i.e. the method names and their signatures, are defined by Pytorch Lightning and are called by Pytorch Lightning,
# and therefore must adhere to the Pytorch Lightning API.
class GenerateTextSamplesCallback(Callback):  # pragma: no cover
    """
    PL Callback to generate triplets along training
    """

    def __init__(self, logging_batch_interval):
        """
        Args:
            logging_batch_interval: How frequently to inspect/potentially plot something
        """
        super().__init__()
        self.logging_batch_interval = logging_batch_interval

    # This is one of the predefined hooks that Pytorch Lightning calls during training. I.e.
    # the name `on_train_batch_end` is not arbitrary; it is a pre-defined name that Pytorch Lightning 
    # will call during training.
    def on_train_batch_end(
        self,
        trainer: Trainer, # The caller of this callback, and the trainer of the model.
        pl_module: LightningModule, # The model being trained. (Name "module" is used because it encapsulates more than just the model, but also the optimizer and other things.)
        outputs: Sequence,
        batch: Sequence, # The input batch, a dictionary of input data (X) and labels (Y)
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # This line of code creates a new table in Weights & Biases (wandb), which is 
        # an experiment tracking and visualization tool commonly used in machine learning projects.
        wandb_table = wandb.Table(columns=["Source", "Pred", "Gold"])
        # pl_module.logger.info("Executing translation callback")
        # This line of code checks if the current batch index is a multiple of the logging_batch_interval.
        # So we are only predicting sample at self.logging_batch_interval, not every batch.
        if (trainer.batch_idx + 1) % self.logging_batch_interval != 0:  # type: ignore[attr-defined]
            return
        labels = batch.pop("labels") # Type of labels is likely torch.Tensor of sequence of token ids.
        # Define generation kwargs to pass to the pl_module.model.generate method, such as the max length
        # of the generated sequence. 
        gen_kwargs = { 
            "max_length": pl_module.hparams.val_max_target_length
            if pl_module.hparams.val_max_target_length is not None
            else pl_module.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "num_beams": pl_module.hparams.eval_beams if pl_module.hparams.eval_beams is not None else pl_module.config.num_beams,
        }
        pl_module.eval() # Set the model to evaluation mode. This is done to temporarily disable dropout and other training-specific layers
                         # and instead use cumulative statistics for normalization layers when making inference, instead of statistics
                         # from just this current batch training, so that we can a more consistent inference/generation. 
                         # After getting the generation, it is important to set the model back to training mode.

        # Important part: This is where we generate the output of the model for the current batch. 

        # Shift the labels to the right to use as decoder inputs
        decoder_inputs = torch.roll(labels, 1, 1)[:,0:2]
        # Then set the first token to 0 (BOS token). In our case `<s>` would be the BOS. 
        decoder_inputs[:, 0] = 0
        # Generate the output of the model for the current batch!
        generated_tokens = pl_module.model.generate(
            batch["input_ids"].to(pl_module.model.device),
            attention_mask=batch["attention_mask"].to(pl_module.model.device),
            decoder_input_ids=decoder_inputs.to(pl_module.model.device),
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded to ensure that the
        # sequences in the batch is of the same length, as torch requires that the tensor is rectangular.  
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = pl_module._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        # Set the model back to training mode since we are done with the generation.
        pl_module.train()

        # Now we decode the predicted output using the tokenizer of the module. 
        # The generated output is a batch/tensor where each row is a 
        # vector/sequence of integer ids of tokens. 
        decoded_preds = pl_module.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # If pad tokens are to be ignored for loss, then replace all labels with the id of -100 
        # with the pad token id. This is because `-100` is a special id that is used to indicate
        # any token to be ignored, and so <pad> would map to this id. 
        # Ignoring any token with id of -100 at loss functioncan be enforced by just not including 
        # these tokens in the loss calculation. The embeddings of these tokens are still
        # computed and used in the forward pass, but they are not included in the loss calculation.
        if pl_module.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = torch.where(labels != -100, labels, pl_module.tokenizer.pad_token_id)

        # Decode the labels and inputs as well.
        decoded_labels = pl_module.tokenizer.batch_decode(labels, skip_special_tokens=False)
        decoded_inputs = pl_module.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)


        # Log the generated samples to the Weights & Biases dashboard.
        # The table is a table of three columns: Source, Pred, Gold.
        # decoded_inputs is decoded X, decoded_preds is the decoded Y_pred, and decoded_labels is the decoded Y_actual
        # pl_module.logger.experiment.log_text('generated samples', '\n'.join(decoded_preds).replace('<pad>', ''))
        # pl_module.logger.experiment.log_text('original samples', '\n'.join(decoded_labels).replace('<pad>', ''))
        for source, translation, gold_output in zip(decoded_inputs, decoded_preds, decoded_labels):
            wandb_table.add_data(
                source.replace('<pad>', ''), translation.replace('<pad>', ''), gold_output.replace('<pad>', '')
            )
        pl_module.logger.experiment.log({"Triplets": wandb_table})
