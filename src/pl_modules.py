from typing import Any
from datetime import datetime
import nltk
import json
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
from score import score, re_score
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from scheduler import get_inverse_square_root_schedule_with_warmup
from datasets import load_dataset, load_metric
from torch.nn.utils.rnn import pad_sequence
from utils import BartTripletHead, shift_tokens_left, extract_triplets_typed, extract_triplets

# A mapping for hyper-parameters argument string, found in `self.hparams.lr_scheduler` to the actual 
# scheduler functions from transformers
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
    "inverse_square_root": get_inverse_square_root_schedule_with_warmup
}
# A mapping for relation type in TACRED dataset to a more readable form
relations_tacred = {'no_relation': 'no relation',
    'org:alternate_names': 'alternate name',
    'org:city_of_branch': 'headquarters location', 
    'org:country_of_branch': 'country of headquarters',
    'org:dissolved': 'dissolved',
    'org:founded_by': 'founded by',
    'org:founded': 'inception',
    'org:member_of': 'member of',
    'org:members': 'has member',
    'org:number_of_employees/members': 'member count',
    'org:political/religious_affiliation': 'affiliation',
    'org:shareholders': 'owned by',
    'org:stateorprovince_of_branch': 'state of headquarters',
    'org:top_members/employees': 'top members',
    'org:website': 'website',
    'per:age': 'age',
    'per:cause_of_death': 'cause of death',
    'per:charges': 'charge',
    'per:children': 'child',
    'per:cities_of_residence': 'city of residence',
    'per:city_of_birth': 'place of birth',
    'per:city_of_death': 'place of death',
    'per:countries_of_residence': 'country of residence',
    'per:country_of_birth': 'country of birth',
    'per:country_of_death': 'country of death',
    'per:date_of_birth': 'date of birth',
    'per:date_of_death': 'date of death',
    'per:employee_of': 'employer',
    'per:identity': 'identity',
    'per:origin': 'country of citizenship',
    'per:other_family': 'relative',
    'per:parents': 'father',
    'per:religion': 'religion',
    'per:schools_attended': 'educated at',
    'per:siblings': 'sibling',
    'per:spouse': 'spouse',
    'per:stateorprovince_of_birth': 'state of birth',
    'per:stateorprovince_of_death': 'state of death',
    'per:stateorprovinces_of_residence': 'state of residence',
    'per:title': 'position held'}
# A mapping for relation type in NYT dataset to a more readable form
relations_nyt = {'/people/person/nationality': 'country of citizenship', '/sports/sports_team/location': 'headquarters location', 
                    '/location/country/administrative_divisions': 'contains administrative territorial entity', '/business/company/major_shareholders': 'shareholders', 
                    '/people/ethnicity/people': 'country of origin', '/people/ethnicity/geographic_distribution': 'denonym', 
                    '/business/company_shareholder/major_shareholder_of': 'major shareholder', '/location/location/contains': 'location',
                    '/business/company/founders': 'founded by', '/business/person/company': 'employer', '/business/company/advisors': 'advisors', 
                    '/people/deceased_person/place_of_death': 'place of death', '/business/company/industry': 'industry', 
                    '/people/person/ethnicity': 'ethnicity', '/people/person/place_of_birth': 'place of birth', 
                    '/location/administrative_division/country': 'country', '/people/person/place_lived': 'residence', 
                    '/sports/sports_team_location/teams': 'member of sports team', '/people/person/children': 'child', 
                    '/people/person/religion': 'religion', '/location/neighborhood/neighborhood_of': 'neighborhood of', 
                    '/location/country/capital': 'capital', '/business/company/place_founded': 'location of formation', 
                    '/people/person/profession': 'occupation'}

# The purpose of this class is to wrap the `transformers` model object to extend its functionalities, so that the model
# and its methods can be integrated with PyTorch Lightning. If we didn't use a wrapper such as pl.LightningModule, then
# the model object (e.g AutoModelForSeq2SeqLM by HuggingFace) can only be cleanly be used with the `transformers` API, but
# not with PyTorch Lightning.  
class BasePLModule(pl.LightningModule):

    # Constructor method of BasePLModule. This class encapsulates and integrates several components necessary for
    # training and inference with PyTorch Lightning:
    # - model: An instance of AutoModelForSeq2SeqLM, typically from the HuggingFace Transformers library.
    # - tokenizer: An instance of AutoTokenizer corresponding to the model, used for tokenizing input texts.
    # - conf: A project-level configuration object managed by Hydra and structured using OmegaConf, containing all necessary hyperparameters and setup configurations.
    # - config: A model-specific configuration (AutoConfig) that includes settings particular to the Transformers model.
    # - *args, **kwargs: Additional arguments and keyword arguments for flexibility and extending functionality.
    # This setup allows the module to be fully compatible with PyTorch Lightning's training and evaluation frameworks, leveraging automatic 
    # optimizations and simplifying the user's code for model management.
    def __init__(self, conf, config: AutoConfig, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.test_results = []
        self.validation_results = []
        if self.model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if self.hparams.label_smoothing == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss
            # Remember, label smoothing is making the OHE of the target label add up to 1,
            # where you make the zeros into epsilons and the one into 1 - epsilons. 
            # This way when computing the cross-entropy loss: 1) model does not learn to 
            # maximize log probability of correct class, and 2) gradient smoothing occurs. 
            self.loss_fn = label_smoothed_nll_loss

    # Perform forward propagation to compute logits and loss. 
    def forward(self, inputs, labels, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        if self.hparams.label_smoothing == 0:
            if self.hparams is not None and self.hparams.ignore_pad_token_for_loss:
                # force training to ignore pad token
                outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
                logits = outputs['logits']
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))#, ignore_index=self.config.pad_token_id)
            else:
                # compute usual loss via models
                outputs = self.model(**inputs, labels=labels, use_cache=False, return_dict = True, output_hidden_states=True)
                loss = outputs['loss']
                logits = outputs['logits']
        else:
            # compute label smoothed loss
            outputs = self.model(**inputs, use_cache=False, return_dict = True, output_hidden_states=True)
            logits = outputs['logits']
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            # labels = torch.where(labels != -100, labels, self.config.pad_token_id)
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            loss, _ = self.loss_fn(lprobs, labels, self.hparams.label_smoothing, ignore_index=self.config.pad_token_id)
        output_dict = {'loss': loss, 'logits': logits}
        # return loss, logits
        return output_dict

    # Perform a single training step, given a batch of data and the batch index.
    # This method processes one batch from the DataLoader during training.
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        labels = batch.pop("labels")
        labels_original = labels.clone()
        # Any label id of -100 is replaced with pad token id, and everything else is left alone. 
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        # Shift the labels to the left by one position and make the last token be a padding token. 
        # Note that for the decoder, the input sequence has to be right shifted. 
        # Here, we are left shifting the labels, and leaving alone the decoder input ids, which
        # is logically equivalent to right shifting the decoder input ids and leaving alone the labels.
        # E.g Here if labels is originally: "<s> Hello world"
        # After left shifting, it becomes: "Hello world <pad>"
        # and decoder_input_ids remains: "<s> Hello world", which is as if we right shifted the decoder input ids.
        labels = shift_tokens_left(labels, -100)
        # Compute the forward pass to get the logits and loss. 
        forward_output = self.forward(batch, labels)
        self.log('loss', forward_output['loss'])
        batch["labels"] = labels_original
        if 'loss_aux' in forward_output:
            self.log('loss_classifier', forward_output['loss_aux'])
            return forward_output['loss'] + forward_output['loss_aux']
        # Return the computed loss of this training step. (we don't need to return the logits
        # since this is training and we don't care about the model's predictions here)
        return forward_output['loss']# + forward_output['loss_aux']

    # Pad input tensors to have maximum length. 
    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                f"Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be padded to `max_length`={max_length}"
            )
        # Basically if input tensor is shape (n, m), then the padded tensor will be of shape (n, max_length), and all
        # values is pad_token_id. 
        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        # Then copy the input tensor to (n, 0:m) of the padded tensor, so that (n: m:max_length) is all pad_token_id.
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    # Here we are going to generate the triplets using the model's forwardpass/inference using the 
    # input `batch` (X), note we don't use `labels` or target (Y) for inference since the target sequence is 
    # autoregressively generated from the BOS token, and iteratively fed to the decoder. Then we 
    # are going to return the decoded predicted triplets and the decoded actual triplets which is 
    # decoded `labels`, as a tuple. 
    # We also use a utility function `extract_triplets_typed` or `extract_triplets` depending on the dataset 
    # to extract the triplets from the decoded text, from a 'linearized triplets' format
    # back into the classical format of ListOf({'head': str, 'type': str, 'tail': str}).
    def generate_triples(self,
        batch,
        labels,
    ) -> None:

        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        # The predicted tokens!
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            use_cache = True,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
        decoded_labels = self.tokenizer.batch_decode(torch.where(labels != -100, labels, self.config.pad_token_id), skip_special_tokens=False)
        if self.hparams.dataset_name.split('/')[-1] == 'conll04_typed.py':
            return [extract_triplets_typed(rel) for rel in decoded_preds], [extract_triplets_typed(rel) for rel in decoded_labels]
        elif self.hparams.dataset_name.split('/')[-1] == 'nyt_typed.py':
            return [extract_triplets_typed(rel, {'<loc>': 'LOCATION', '<org>': 'ORGANIZATION', '<per>': 'PERSON'}) for rel in decoded_preds], [extract_triplets_typed(rel, {'<loc>': 'LOCATION', '<org>': 'ORGANIZATION', '<per>': 'PERSON'}) for rel in decoded_labels]
        elif self.hparams.dataset_name.split('/')[-1] == 'docred_typed.py':
            return [extract_triplets_typed(rel, {'<loc>': 'LOC', '<misc>': 'MISC', '<per>': 'PER', '<num>': 'NUM', '<time>': 'TIME', '<org>': 'ORG'}) for rel in decoded_preds], [extract_triplets_typed(rel, {'<loc>': 'LOC', '<misc>': 'MISC', '<per>': 'PER', '<num>': 'NUM', '<time>': 'TIME', '<org>': 'ORG'}) for rel in decoded_labels]
        # Return type is ([[triplet1, ...], ...],[[triplet1, ...], ...])
        return [extract_triplets(rel) for rel in decoded_preds], [extract_triplets(rel) for rel in decoded_labels]

    # Given the firt triplet, predict the subsequent triplets. 
    def generate_samples(self,
        # model,
        # tokenizer,
        batch,
        labels,
    ) -> None:
        # labels = batch.pop("labels")
        # pick the last batch and logits
        # x, y = batch
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "length_penalty": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        # token id 50265 is for `<obj>` which essentially marks position of the triplet.
        # E.g. [<triplet>, h1, <subj>, t1, <obj>, r1, <subj>, t2, <obj>, r2] -> [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        relation_start = labels == 50265 

        # [0, 0, 0, 0, 1, 0, 0, 0, 1, 0] -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        relation_start = torch.roll(relation_start, 1, 1) 

        # [0, 0, 0, 0, 0, 1, 0, 0, 0, 1] -> [0, 0, 0, 0, 0, 1, 1, 1, 1, 2]
        relation_start = torch.cumsum(relation_start, dim=1)

        # [<triplet>, h1, <subj>, t1, <obj>, r1, <subj>, t2, <obj>, r2] -> 
        # [<triplet>, h1, <subj>, t1, <obj>, <pad>, <pad>, <pad>, <pad>, r2] 
        labels_decoder = torch.where(relation_start == 1, self.tokenizer.pad_token_id, labels)

        # [<triplet>, h1, <subj>, t1, <obj>, <pad>, <pad>, <pad>, <pad>, </s>] 
        labels_decoder[:,-1] = 2 # token id 2 is for `</s>`
        
        # [</s>, <triplet>, h1, <subj>, t1, <obj>, <pad>, <pad>, <pad>, <pad>]
        labels_decoder = torch.roll(labels_decoder, 1, 1)

        # The predicted tokens!
        generated_tokens = self.model.generate(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device), 
            use_cache = False,
            **gen_kwargs,
        )
        # Get a boolean mask tensor, again, for the `<obj>` token.
        relation_start = generated_tokens == 50265 
        # Circular roll to the right by 2 positions.
        relation_start = torch.roll(relation_start, 2, 1)

        # From the generated tokens select just the tokens following the `<obj>` token, and decode them to text.
        # So we are effectively just extracting the relation types from the generated triplets. 
        decoded_preds = self.tokenizer.batch_decode(generated_tokens[relation_start==1], skip_special_tokens=False)

        # For each relation type, strip any leading or trailing whitespaces.
        return [rel.strip() for rel in decoded_preds]

    def forward_samples(self, batch, labels) -> None:
        relation_start = labels == 50265
        relation_start = torch.roll(relation_start, 2, 1)
        labels = torch.where(torch.cumsum(relation_start, dim=1) == 1, self.tokenizer.pad_token_id, labels)
        labels[:,-1] = 0
        labels = torch.roll(labels, 1, 1)
        min_padding = min(torch.sum((labels == 1).int(), 1))
        labels_decoder = torch.randint(60000,(labels.shape[0], labels.shape[1] - min_padding))
        labels_decoder = labels[:, :-min_padding]

        labels_decoder = torch.where(labels_decoder != -100, labels_decoder, self.config.pad_token_id)

        outputs = self.model(
            batch["input_ids"].to(self.model.device),
            attention_mask=batch["attention_mask"].to(self.model.device),
            decoder_input_ids=labels_decoder.to(self.model.device), # This is for 'teacher-forcing' i.e. decoder_input_ids is the guide.
            return_dict=True,
        )

        next_token_logits = outputs.logits[relation_start[:,: -min_padding]==1]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        decoded_preds = self.tokenizer.batch_decode(next_tokens, skip_special_tokens=False)

        return [rel.strip() for rel in decoded_preds]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = batch.pop("labels")
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)
        labels = shift_tokens_left(labels, -100)
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        forward_output['loss'] = forward_output['loss'].mean().detach()

        if self.hparams.prediction_loss_only:
            self.log('val_loss', forward_output['loss'])
            return

        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels

        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}
        metrics['val_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key])

        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect each batch's outputs
        self.validation_results.append(outputs)
    
    def on_validation_epoch_end(self) -> Any:
        outputs = self.validation_results
        if self.hparams.relations_file:
            relations_df = pd.read_csv(self.hparams.relations_file, header = None, sep='\t')
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], relations)
            self.log('val_prec_micro', precision)
            self.log('val_recall_micro', recall)
            self.log('val_F1_micro', f1)
        # elif not 'tacred' in self.hparams.dataset_name.split('/')[-1]:
        #     if self.hparams.dataset_name.split('/')[-1] == 'conll04_typed.py':
        #         scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], ['killed by', 'residence', 'location', 'headquarters location', 'employer'], "strict")
        #     elif self.hparams.dataset_name.split('/')[-1] == 'ade.py':
        #         scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], ['has effect'])
        #     elif self.hparams.dataset_name.split('/')[-1] == 'nyt_typed.py':
        #         scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], list(relations_nyt.values()), "strict")
        #     elif self.hparams.dataset_name.split('/')[-1] == 'docred_typed.py':
        #         relations_docred = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}
        #         scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], list(relations_docred.values()), "strict")            
        #     else:
        #         scores, precision, recall, f1 = re_score([item for pred in output for item in pred['predictions']], [item for pred in output for item in pred['labels']], ['killed by', 'residence', 'location', 'headquarters location', 'employer'])
        #     self.log('val_prec_micro', precision)
        #     self.log('val_recall_micro', recall)
        #     self.log('val_F1_micro', f1)
        else:
            preds_list = []
            labels_list = []
            for batch_output in outputs:
                for pred, lab in zip(batch_output['predictions'], batch_output['labels']):
                    if len(pred) == 0 or len(lab) == 0:
                        continue
                    preds_list.append(pred[0]["type"])
                    labels_list.append(lab[0]["type"])
            prec_micro, recall_micro, f1_micro = score(labels_list, preds_list, verbose=True)
            self.log('val_prec_micro', prec_micro)
            self.log('val_recall_micro', recall_micro)
            self.log('val_F1_micro', f1_micro)

    # This method is called by PyTorch Lightning for every batch of data in the test dataset. 
    # The operations you've described inside this method, such as generating tokens, calculating loss, 
    # and optionally computing metrics, are performed here. The method handles how each individual 
    # test batch should be processed. The return values from this method are collected and can be 
    # utilized in the subsequent steps or for metrics calculation.
    def test_step(self, batch: dict, batch_idx: int) -> None:
        gen_kwargs = {
            "max_length": self.hparams.val_max_target_length
            if self.hparams.val_max_target_length is not None
            else self.config.max_length,
            "early_stopping": False,
            "no_repeat_ngram_size": 0,
            "length_penalty": 0,
            "num_beams": self.hparams.eval_beams if self.hparams.eval_beams is not None else self.config.num_beams,
        }

        if self.hparams.predict_with_generate and not self.hparams.prediction_loss_only:
            generated_tokens = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        # Get the target sequence from batch.
        labels = batch.pop("labels")
        # Note that the both the input and target labels are preprocessed by tokenizer such that they are both
        # tagged with <s> BOS token. Try below to see.
        # first_token_ids = labels[:, 0]  # Extract the first token ID from each sequence
        # decoded_tokens = [self.tokenizer.decode([token_id]) for token_id in first_token_ids]
        # print(decoded_tokens)

        # Replace any label id of -100 with pad token id, and leave everything else as is.
        batch["decoder_input_ids"] = torch.where(labels != -100, labels, self.config.pad_token_id)

        # Then left shift target labels by 1, so that the prediction aligns with the target,
        # since the prediction are the next tokens after the BOS token, it doesn't include BOS,
        # so the target should also not include BOS.
        # More specifically, when computing the loss of the model in the decoder phase, the input 
        # to decoder is output of encoder and just the BOS token.
        # Since the target is always the next token, the target has to be one left-shifted with 
        # respect to the autoregressively generated input. (or the autoregressively generated input
        # is right-shifted with respect to the target, same thing). 
        # I.e. If the labels is "<s> Hello world", then after left shifting, it becomes "Hello world <pad>"
        # and the decoder should, starting from <s>, autoregressively generate the next tokens until </s>,
        # e.g. "Hi", "world", </s>. So then the prediction becomes "Hi world </s>", and the target is "Hello world </s>".
        # So we can then compare "Hi" with "Hello", "world" with "world", and "</s>" with "</s>" to compute the loss.
        labels = shift_tokens_left(labels, -100) 

        # And we compute the forward pass to get the logits and loss.
        with torch.no_grad():
            # compute loss on predict data
            forward_output = self.forward(batch, labels)

        # If we only want to get the loss from test set, then we log the loss and return.
        forward_output['loss'] = forward_output['loss'].mean().detach()
        if self.hparams.prediction_loss_only:
            self.log('test_loss', forward_output['loss'])
            return

        # Else, use the logits (i.e. raw predictions) and the target labels to compare them and compute the metrics.
        # detach() is a pytorch method that detaches the tensor from the computation graph, so that the tensor is not
        # tracked for gradient computation. 
        forward_output['logits'] = generated_tokens.detach() if self.hparams.predict_with_generate else forward_output['logits'].detach()

        if labels.shape[-1] < gen_kwargs["max_length"]:
            forward_output['labels'] = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            forward_output['labels'] = labels
        if self.hparams.predict_with_generate:
            metrics = self.compute_metrics(forward_output['logits'].detach().cpu(), forward_output['labels'].detach().cpu())
        else:
            metrics = {}
        # And of course add the test loss to the metrics as well for more detailed evaluation.
        metrics['test_loss'] = forward_output['loss']
        for key in sorted(metrics.keys()):
            self.log(key, metrics[key], prog_bar=True)
        # This is the end of logging the test set evaluation metrics for this test batch.

        # And finally, more importantly, we want to generate the triplets from the model's predictions
        # so that we can actually see the predicted triplets sequences.
        # if self.hparams.finetune:
        #     return {'predictions': self.forward_samples(batch, labels)}
        # else:
        #     outputs = {}
        #     outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        #     return outputs
        outputs = {}
        outputs['predictions'], outputs['labels'] = self.generate_triples(batch, labels)
        return outputs
        
    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect each batch's outputs
        self.test_results.append(outputs)

    # After all batches have been processed through test_step, this method is called. 
    # It's used to summarize or aggregate the results collected over the epoch, such 
    # as computing overall metrics from the outputs of the test steps. This method 
    # provides a hook to perform actions that consider the entire dataset, such as 
    # calculating precision, recall, and F1 scores across all test data.
    def on_test_epoch_end(self) -> Any:
        outputs = self.test_results
        if not self.hparams.finetune and self.hparams.relations_file:
            print(f'\n\nTesting results for `{self.hparams.model_name_or_path}` model which is not fine-tuned.' 
                f' Only considering the relations in the relations file `{self.hparams.relations_file}`')
            relations_df = pd.read_csv(self.hparams.relations_file, header = None, sep='\t')
            relations = list(relations_df[0])
            scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], relations)
            self.log('test_prec_micro', precision)
            self.log('test_recall_micro', recall)
            self.log('test_F1_micro', f1)
        # elif not 'tacred' in self.hparams.dataset_name.split('/')[-1]:
        #     if self.hparams.dataset_name.split('/')[-1] == 'conll04_typed.py':
        #         scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], ['killed by', 'residence', 'location', 'headquarters location', 'employer'], "strict")
        #     elif self.hparams.dataset_name.split('/')[-1] == 'ade.py':
        #         scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], ['has effect'])
        #     elif self.hparams.dataset_name.split('/')[-1] == 'nyt_typed.py':
        #         scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], list(relations_nyt.values()), "strict")
        #     elif self.hparams.dataset_name.split('/')[-1] == 'docred_typed.py':
        #         relations_docred = {"P6": "head of government", "P17": "country", "P19": "place of birth", "P20": "place of death", "P22": "father", "P25": "mother", "P26": "spouse", "P27": "country of citizenship", "P30": "continent", "P31": "instance of", "P35": "head of state", "P36": "capital", "P37": "official language", "P39": "position held", "P40": "child", "P50": "author", "P54": "member of sports team", "P57": "director", "P58": "screenwriter", "P69": "educated at", "P86": "composer", "P102": "member of political party", "P108": "employer", "P112": "founded by", "P118": "league", "P123": "publisher", "P127": "owned by", "P131": "located in the administrative territorial entity", "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains administrative territorial entity", "P155": "follows", "P156": "followed by", "P159": "headquarters location", "P161": "cast member", "P162": "producer", "P166": "award received", "P170": "creator", "P171": "parent taxon", "P172": "ethnic group", "P175": "performer", "P176": "manufacturer", "P178": "developer", "P179": "series", "P190": "sister city", "P194": "legislative body", "P205": "basin country", "P206": "located in or next to body of water", "P241": "military branch", "P264": "record label", "P272": "production company", "P276": "location", "P279": "subclass of", "P355": "subsidiary", "P361": "part of", "P364": "original language of work", "P400": "platform", "P403": "mouth of the watercourse", "P449": "original network", "P463": "member of", "P488": "chairperson", "P495": "country of origin", "P527": "has part", "P551": "residence", "P569": "date of birth", "P570": "date of death", "P571": "inception", "P576": "dissolved, abolished or demolished", "P577": "publication date", "P580": "start time", "P582": "end time", "P585": "point in time", "P607": "conflict", "P674": "characters", "P676": "lyrics by", "P706": "located on terrain feature", "P710": "participant", "P737": "influenced by", "P740": "location of formation", "P749": "parent organization", "P800": "notable work", "P807": "separated from", "P840": "narrative location", "P937": "work location", "P1001": "applies to jurisdiction", "P1056": "product or material produced", "P1198": "unemployment rate", "P1336": "territory claimed by", "P1344": "participant of", "P1365": "replaces", "P1366": "replaced by", "P1376": "capital of", "P1412": "languages spoken, written or signed", "P1441": "present in work", "P3373": "sibling"}
        #         scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], list(relations_docred.values()), "strict")            
        #     else:
        #         scores, precision, recall, f1 = re_score([item for pred in outputs for item in pred['predictions']], [item for pred in outputs for item in pred['labels']], ['killed by', 'residence', 'location', 'headquarters location', 'employer'])
        #     self.log('test_prec_micro', precision)
        #     self.log('test_recall_micro', recall)
        #     self.log('test_F1_micro', f1)
        else:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            prediction_file = current_time + '_preds.jsonl'
            if self.hparams.finetune:
                print(f'\n\nTesting results for `{self.hparams.checkpoint_path}` model.' 
                    f' The test file is `{self.hparams.test_file}` and'
                    f' the prediction result is in the file `{prediction_file}`')
            else:
                print(f'\n\nTesting results for `{self.hparams.model_name_or_path}` model.' 
                    f' The test file is `{self.hparams.test_file}` and'
                    f' the prediction result is in the file `{prediction_file}`')
            # key = []
            # with open(self.hparams.test_file) as json_file:
            #     f = json.load(json_file)
            #     for id_, row in enumerate(f):
            #         key.append(' '.join(row['token']))
            with open(prediction_file, 'w') as f:
                f.write('Model name: ' + self.hparams.model_name_or_path + '\n')
                f.write('Test file: ' + self.hparams.test_file + '\n')
                f.write('predictions \t labels \n')
                preds_list = []
                labels_list = []
                for ele in outputs:
                    for pred, lab in zip(ele['predictions'], ele['labels']):
                        if len(pred) == 0 or len(lab) == 0:
                            continue
                        # We are just using the first triplet in case there are multiple triplets predicted.
                        f.write(f'{pred[0]} \t {lab[0]} \n')
                        preds_list.append(pred[0]["type"])
                        labels_list.append(lab[0]["type"])

            prec_micro, recall_micro, f1_micro = score(labels_list, preds_list, verbose=True)
            self.log('test_prec_micro', prec_micro)
            self.log('test_recall_micro', recall_micro)
            self.log('test_F1_micro', f1_micro)

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.hparams.adafactor else AdamW
        if self.hparams.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.hparams.adam_beta1, self.hparams.adam_beta2),
                "eps": self.hparams.adam_epsilon,
            }
        
        optimizer_kwargs["lr"] = self.hparams.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = self._get_lr_scheduler(self.hparams.max_steps, optimizer)

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_lr_scheduler(self, num_training_steps, optimizer):
        schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = schedule_func(optimizer)
        elif self.hparams.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "inverse_square_root":
            # args = {"warmup_updates": self.hparams.warmup_steps, "lr": [self.hparams.learning_rate]}
            scheduler = schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = schedule_func(
                optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler


    def compute_metrics(self, preds, labels):
        metric_name = "rouge" # if self.hparams.task.startswith("summarization") else "sacrebleu"
        metric = load_metric(metric_name)
        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            # rougeLSum expects newline after each sentence
            if metric_name == "rouge":
                preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
                labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
            else:  # sacrebleu
                labels = [[label] for label in labels]

            return preds, labels
        # preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.hparams.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        if metric_name == "rouge":
            result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
            # Extract a few results from ROUGE
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

