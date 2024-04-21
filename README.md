# Introduction:

This is the source code for final project: 'Biomedical Knowledge Graph Generation by
fine-tuning REBEL model'

This is a forked repo of: `https://github.com/Babelscape/rebel`
and thus all credits for the project structure and the REBEL model goes to the authors of the 
source research paper: 
```
@inproceedings{huguet-cabot-navigli-2021-rebel-relation,
    title = "{REBEL}: Relation Extraction By End-to-end Language generation",
    author = "Huguet Cabot, Pere-Llu{\'\i}s  and
      Navigli, Roberto",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.204",
    pages = "2370--2381",
    abstract = "Extracting relation triplets from raw text is a crucial task in Information Extraction, enabling multiple applications such as populating or validating knowledge bases, factchecking, and other downstream tasks. However, it usually involves multiple-step pipelines that propagate errors or are limited to a small number of relation types. To overcome these issues, we propose the use of autoregressive seq2seq models. Such models have previously been shown to perform well not only in language generation, but also in NLU tasks such as Entity Linking, thanks to their framing as seq2seq tasks. In this paper, we show how Relation Extraction can be simplified by expressing triplets as a sequence of text and we present REBEL, a seq2seq model based on BART that performs end-to-end relation extraction for more than 200 different relation types. We show our model{'}s flexibility by fine-tuning it on an array of Relation Extraction and Relation Classification benchmarks, with it attaining state-of-the-art performance in most of them.",
}
```

# Installation

The origianl repo used dated PyTorch Lightning versions whose source codes changed a lot in recent versions.
We used Python 3.11, and for most packages, we used the most up-to-date versions at the time of 2024-April.
To install the dependencies, run:

`pip install -r requirements.txt`

# Reproducing the fine-tuning and evaluation

## Fine-tune REBEL on BioRel

1. Go to `conf/model/default_model.yaml` and set these parameters:
```
model_name_or_path: 'Babelscape/rebel-large'
config_name: 'Babelscape/rebel-large'
tokenizer_name: 'Babelscape/rebel-large' 
finetune: True
```

2. Go to `conf/data/default_data.yaml` and set these parameters:
```
train_file: 'data/biorel/train.jsonl'
validation_file: 'data/biorel/valid.jsonl'
test_file: 'data/biorel/test.jsonl'
```

3. Then run this command to finetune REBEL on BioRel train dataset:

`python src/train.py`

Or if you want to train the base REBEL model on a computing cluster that uses the `Slurm` job scheduler, run the following:

`bash finetune_model_slurm.sh` 

## Test fine-tuned model

To test the test data split of BioRel, run the following:

`python src/test.py`

Or again, if you want to run on a computing cluster:

`bash test_model_slurm.sh`

# Run the model on sample text input

If you want to see how the model works on a sample input text, you can do so interactively using the below
demo Jupyter Notebook:

`demo/demo.ipynb`

Running all the cells, it should generate a `<some name>.html` file which is a simple graphical 
representation of your knowledge graph. 

