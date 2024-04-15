from typing import Any, Union, List, Optional
import os
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasets import load_dataset, set_caching_enabled
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    default_data_collator,
    set_seed,
)

# This class is a PyTorch Lightning DataModule that is used to create DataLoader object. The DataLoader object, the object which
# this class outputs, is the format of data that the Pytorch Lightning's Trainer object requires. 
# Without a DataLoader object, we have to do the batching, shuffling, tensorization, hardware configuration manually, which 
# is a lot of work - DataLoader encapsulates all this into a single object.

# Similar to how a subclass of datasets.BuilderConfig implements the preset interface methods, and these methods are automatically
# triggered by the datasets.load_dataset() function, the mechanism is similar.

# In PyTorch Lightning, the methods in LightningDataModule such as prepare_data, setup, train_dataloader, val_dataloader, 
# and test_dataloader are automatically triggered by the PyTorch Lightning Trainer at specific points in the training lifecycle. 
# The developer does not need to manually call these methods; instead, they define these methods in their LightningDataModule 
# subclass, and then simply pass an instance of this subclass to the Trainer. The Trainer takes care of invoking these methods 
# at the appropriate times, ensuring the data is prepared and loaded correctly for the training, validation, and testing phases. 
# This automation helps streamline the workflow and makes the code cleaner and more modular.

# The BasePLDataModule takes the dataset (or DatasetDict) that has already been processed by the Ade builder or similar and 
# wraps it in a structure that integrates well with PyTorch Lightning workflows for training, validation, and testing.
# I.e. the datasets.GeneratorBasedBuilder was used to create the dataset, i.e. the DatasetDict object.
# The pl.LightningDataModule is then used to wrap this DatasetDict with other information and outputs DataLoader object via
# the methods train_dataloader, val_dataloader, and test_dataloader. 

# Remember that the point of DataLoader is to automize a pipeline on how to apply additional transformations and process 
# the data by the Trainor object. Some processes it automizes are shuffling, batching, and loading the data in parallel.
# The PyTorch Lightning Trainer object requires the dataset to be provided through a DataLoader, not directly as a Dataset 
# or DatasetDict object. It is an additional layer of abstraction for precisely formatting the input for the training process.

# DatasetDict: Data management into splits, and preprocessing of raw data into a more usable format but still operating at human
# readable data level. 
# DataLoader: Data management into batches, shuffling, hardware configuration (like are we using CPU or GPU, how many 
# workers are we using), and processing of data into a computer readable format (tensors) that can be fed into, and 
# processed by, the model. Note, 'data collator' is the component of DataLoader that processes the data into a format that 
# can be fed into the model.
class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """
    # Create the BasePLDataModule which contains: Hydra configuration, tokenizer, and model,
    # dataset (DatasetDict object), and data collator object.
    def __init__(self, conf: DictConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.model = model

        # Just to get the absolute path of the dataset files, where the paths in the
        # yaml config is relative path, but we need absolute path to load the dataset.
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_name_abs_path = str(os.path.join(base_path, conf.dataset_name))
        self.train_file_abs_path = str(os.path.join(base_path, conf.train_file))
        self.validation_file_abs_path = str(os.path.join(base_path, conf.validation_file))
        self.test_file_abs_path = str(os.path.join(base_path, conf.test_file))

        if conf.relations_file:
            self.relations_file_abs_path = str(os.path.join(base_path, conf.relations_file))
        else:
            self.relations_file_abs_path = ""

        if conf.relations_file:
            self.datasets = load_dataset(self.dataset_name_abs_path, data_files={'train': self.train_file_abs_path, 
                                                                        'dev': self.validation_file_abs_path, 
                                                                        'test': self.test_file_abs_path, 
                                                                        'relations': self.relations_file_abs_path})
        else:
            self.datasets = load_dataset(self.dataset_name_abs_path, data_files={'train': self.train_file_abs_path, 
                                                                        'dev': self.validation_file_abs_path, 
                                                                        'test': self.test_file_abs_path})
        set_caching_enabled(True)
        self.prefix = conf.source_prefix if conf.source_prefix is not None else ""
        self.column_names = self.datasets["train"].column_names
        # self.source_lang, self.target_lang, self.text_column, self.summary_column = None, None, None, None
        self.text_column = conf.text_column
        self.summary_column = conf.target_column
        self.max_target_length = conf.max_target_length
        self.padding = "max_length" if conf.pad_to_max_length else False

        # Data collator
        label_pad_token_id = -100 if conf.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if conf.pad_to_max_length:
            self.data_collator = default_data_collator
        else:
            self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model, label_pad_token_id=label_pad_token_id)

    # Apply the computer level preprocessing to our datset from the DatasetDict object, i.e. we are not
    # "tensorizing" the data by applying tokenizer. And also applying other low level hardware level optimization 
    # stuff like caching,
    def prepare_data(self, *args, **kwargs):
        self.train_dataset = self.datasets["train"]
        if "train" not in self.datasets:
            raise ValueError("--do_train requires a train dataset")
        if self.conf.max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(range(self.conf.max_train_samples))
        # This is the core part of preprocessing. We are mapping the dataset to the `preprocess_function`
        # by using `Dataset.map()`. (note DatsetDict['train'] is a Dataset object, and Dataset.map() is a
        #  method of Dataset object). The additional keyword arguments help customize and optimize how this 
        # operation is performed for faster processing. 
        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            num_proc=self.conf.preprocessing_num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=not self.conf.overwrite_cache,
            cache_file_name=self.train_file_abs_path.replace('.jsonl', '-') + self.dataset_name_abs_path.split('/')[-1].replace('.py', '.cache'),
        )

        if self.conf.do_eval:
            max_target_length = self.conf.val_max_target_length
            if "validation" not in self.datasets:
                raise ValueError("--do_eval requires a validation dataset")
            self.eval_dataset = self.datasets["validation"]
            if self.conf.max_val_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(self.conf.max_val_samples))
            self.eval_dataset = self.eval_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=self.validation_file_abs_path.replace('.jsonl', '-') + self.dataset_name_abs_path.split('/')[-1].replace('.py', '.cache'),
            )

        if self.conf.do_predict:
            max_target_length = self.conf.val_max_target_length
            if "test" not in self.datasets:
                raise ValueError("--do_predict requires a test dataset")
            self.test_dataset = self.datasets["test"]
            if self.conf.max_test_samples is not None:
                self.test_dataset = self.test_dataset.select(range(self.conf.max_test_samples))
            self.test_dataset = self.test_dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=self.conf.preprocessing_num_workers,
                remove_columns=self.column_names,
                load_from_cache_file=not self.conf.overwrite_cache,
                cache_file_name=self.test_file_abs_path.replace('.jsonl', '-') + self.dataset_name_abs_path.split('/')[-1].replace('.py', '.cache'),
            )

    # This method is triggered to return the DataLoader object for training, after the training data has been prepared. 
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.train_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
            shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.conf.eval_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.conf.eval_batch_size,
            # sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.conf.dataloader_drop_last,
            num_workers=self.conf.dataloader_num_workers,
            pin_memory=self.conf.dataloader_pin_memory,
        )

    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     raise NotImplementedError

    # Function to preprocess the 3data by encoding it into tensor object using the tokenizer. 
    # This function is called by the `prepare_data` method to map it to the dataset object.
    # Here we are doing: prefix adding, tokenization, padding, and truncation.
    def preprocess_function(self, examples):
        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenizer(inputs, max_length=self.conf.max_source_length, padding=self.padding, truncation=True)

        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss. `self.padding = "max_length"` is a padding strategy to pad until max_length is reached
        # if sequence is less than max_length.
        if self.padding == "max_length" and self.conf.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        # model_inputs["decoder_input_ids"] = labels["input_ids"]
        # model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        # model_inputs["labels"] = shift_tokens_left(labels["input_ids"], self.tokenizer.pad_token_id)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs # A dict whose keys are: input_ids, attention_mask, labels