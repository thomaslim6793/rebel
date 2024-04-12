# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""ADE"""

# __future__ is a special library in Python that allows you to use features from future versions of Python.
from __future__ import absolute_import, division, print_function 

import json
import logging

import datasets

import math
# defauldict is a dict subclass that sets up a default value for a key that does not exists.
from collections import defaultdict

_DESCRIPTION = """\
ADE dataset.
"""

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

mapping = {'Adverse-Effect': 'has effect'}

class AdeConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        # initialize the super class attributes by passing **kwargs to the __init__ method of the super class.
        # Here the class itself, i.e. `AdeConfig` is used as the first argument in super(), which is related to how 
        # Python's method resolution order (MRO) works. When you call super() with the class itself and an 
        # instance (self), Python uses the MRO to find the next class in line after AdeConfig to look for the method to call. 
        # This is typically the immediate parent class of AdeConfig, but in more complex inheritance scenarios, it could be 
        # another class in the hierarchy.
        super(AdeConfig, self).__init__(**kwargs)

# Description: A datasets.GeneratorBasedBuilder subclass that defines how to load and preprocess the ADE dataset. 
# It is a subclass of datasets.GeneratorBasedBuilder of the HuggingFace `datasets` library. 
# This class is called/instantiated internally by `datasets` functions; it is not directly called by the developer. 
# For the class to be compatible with the datasets library, it must implement the following methods: 
# _info, _split_generators, and _generate_examples.
#
# Ultimately, calling `dataset.load_dataset` method calls/instantiates this class, and calls its 3 methods, to build the 
# `DatasetDict` object which is returned to the user.
# This class is called implicitly by doing this:
# >> my_dataset = load_dataset('ade.py', 'plain_text')  
class Ade(datasets.GeneratorBasedBuilder):
    """Ade Version 1.0."""
    # This is a class attribute, accessed independently of any object of this class. I.e. Ade.BUILDER_CONFIGS is used. 
    BUILDER_CONFIGS = [
        AdeConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    # Provides metadata about the ADE dataset. 
    # How this method is called: Called by the `load_dataset` function 
    # Its return value is accessed as follows:
    # >> my_dataset = load_dataset('ade.py', 'plain_text')
    # >> my_dataset.info 
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            # Specifies the data types of the features in the dataset.
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://www.sciencedirect.com/science/article/pii/S1532046412000615?via%3Dihub/",
        )

    # Description: This method downloads the data and defines the splits of the dataset.
    # How this method is called: Called by the `load_dataset` function when the user loads the dataset.
    # Its return value is accessed as follows:
    # >> my_dataset = load_dataset('ade', 'plain_text')
    # >> my_dataset['train'], my_dataset['validation'], my_dataset['test']
    def _split_generators(self, dl_manager):
        # If the config has data_files, then use those files. Otherwise, download the files.
        # When you load a dataset using the load_dataset function, you can optionally provide a data_files argument to specify 
        # the paths to your local data files for each split of the dataset. This argument is then stored in the config attribute 
        # of the dataset builder instance (self.config in your class).
        # Doing the following assigns value to the self.config.data_files attribute:
        # >> my_dataset = load_dataset('ade', 'plain_text', data_files={'train': 'path/to/train.json', 'dev': 'path/to/dev.json', 'test': 'path/to/test.json'})
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], 
                "dev": self.config.data_files["dev"], 
                "test": self.config.data_files["test"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)
        # Return the train, validation, and test split objects as a list. These three objects are of the 
        # class datasets.SplitGenerator.
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    # Description: This method processes the data into the standard format of training data for the model.
    # The standard format has four columns: 'id', 'title', 'context', 'triplets'.
    # The two key columns are: 'context' and 'triplets'.
    # Input: 'context' 
    # Target: 'triplets' 
    # The crucial preprocessing is done to the 'triplets' column, where the target value of data is triplets in a non-standardized format.
    # We want all datasets to adhere to the same standard format of triplets target values. The authors of REBEL defined the 
    # 'linearized triplets' format as the standard format for triplets.
    #
    # Non-standardized formats of triplets: 
    # Non-standardized format 1: {head: [entity 1, entity 1] tail: [entity 2, entity 3], relation type: [relation type 1, relation type 2]}
    # Non-standardized format 2: {{head: entity1, tail: entity2, relation type: relation type 1}, 
    #                             {head: entity1, tail: entity3, relation type: relation type 2}}
    # Non-standardized format 3: {head: entity1, tail: entity2, relation type: relation type 1}
    # Standardized format, a.k.a. 'linearized triplets' format:
    # "<triplets> entity 1 <subj> entity 2 <obj> relation type 1 <subj> entity 3 <obj> relation type 2"
    def _generate_examples(self, filepath):
        logging.info("generating examples from = %s", filepath)

        with open(filepath) as json_file:
            f = json.load(json_file)
            for id_, row in enumerate(f):
                triplets = ''
                prev_head = None
                for relation in row['relations']:
                    # If previous head is the same as the current head, then append just the tail and relation-type to the current triplet. 
                    if prev_head == relation['head']:
                        triplets += (f' <subj> ' + 
                                     ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + 
                                     f' <obj> ' + mapping[relation['type']])
                    # If the previous head is none, then start a new triplet with the head, tail, and relation-type, but no white space before 
                    # the `<triplet>`` tag. 
                    elif prev_head == None:
                        triplets += ('<triplet> ' + 
                                     ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' <subj> ' + 
                                     ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' <obj> ' 
                                     + mapping[relation['type']])
                        prev_head = relation['head']
                    # If the previous head is not the same as the current head, then start a new triplet with the head, tail, and relation-type, but 
                    # with white space before the `<triplet>`` tag.
                    else:
                        triplets += (' <triplet> ' + 
                                     ' '.join(row['tokens'][row['entities'][relation['head']]['start']:row['entities'][relation['head']]['end']]) + f' <subj> ' + 
                                     ' '.join(row['tokens'][row['entities'][relation['tail']]['start']:row['entities'][relation['tail']]['end']]) + f' <obj> ' + 
                                     mapping[relation['type']])
                        prev_head = relation['head']
                text = ' '.join(row['tokens'])
                # When you use a yield statement in a function, it turns the function into a generator, and it pauses the function and saves its state until 
                # the next time next() is called. When next() is called, the function resumes executing immediately after the yield statement.
                # The whole point is that, if the below tuple is collected in a list and returned, then the function basically has to do all the preprocessing
                # for all rows and then return the object to the user. This is clearly memory dependent. But with the `yield` statement at the end of the loop,
                # we can execute an iteration of this for-loop and return the value of this `yield` statement "on-demand". Simply put, lazy evaluation.
                # Note we can implement this using Python iterator of which generator is a subclass of, but using generator (i.e. using yield) is syntactically
                # much more concise and readable. This of generator as a special syntax that Python "compiles" at runtime into
                # a class with iterator protocol, i.e. having the `__iter__` and `__next__` method.
                yield str(row["orig_id"]), {
                    "title": str(row["orig_id"]),
                    "context": text,
                    "id": str(row["orig_id"]),
                    "triplets": triplets,
                }