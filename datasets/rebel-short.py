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
"""REBEL"""

from __future__ import absolute_import, division, print_function

import pandas as pd

import datasets

import re 
import json
import logging
import math
from collections import defaultdict

_DESCRIPTION = """\
REBEL is a silver dataset created for the paper REBEL: Relation Extraction By End-to-end Language generation
"""

_URL = ""
_URLS = {
    "train": _URL + "en_train.jsonl",
    "dev": _URL + "en_val.jsonl",
    "test": _URL + "en_test.jsonl",
}


class RebelConfig(datasets.BuilderConfig):
    """BuilderConfig for REBEL."""

    def __init__(self, **kwargs):
        """BuilderConfig for REBEL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RebelConfig, self).__init__(**kwargs)


class Rebel(datasets.GeneratorBasedBuilder):
    """Rebel 1.0"""

    BUILDER_CONFIGS = [
        RebelConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
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
            # homepage="",
#             citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"], #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"], #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)
        relations_df = pd.read_csv(self.config.data_files['relations'], header = None, sep='\t')
        relations = list(relations_df[0])

        with open(filepath, encoding="utf-8") as f:
            for id_, row in enumerate(f):
                article = json.loads(row)
                prev_len = 0
                if len(article['triples']) == 0:
                    continue
                count = 0
                for text_paragraph in article['text'].split('\n'):
                    if len(text_paragraph) == 0:
                        continue
                    sentences = re.split(r'(?<=[.])\s', text_paragraph)
                    text = ''
                    for sentence in sentences:
                        # For each entity in article's entities, if any of the entity boundary starts before the current sentence and ends after the current sentence, skip the sentence.
                        # This only really happens in extremely rare edge cases where the named entity contains ". " substring in the middle of it. 
                        # For example: "I love Google Inc. Robotics Division because it is the best." and if the entity is "Google Inc. Robotics Division", then this sentence will
                        # actually be split as two sentences "I love Google Inc." "Robotics Division because it is the best.", and the below condition will be true, causing 
                        # the sentence to be skipped. 
                        # So this check basically ignores incorrect sentence splits, and the `text` accumulator is designed to concatenate this incorrect
                        # sentence splits back together. 
                        # When a sentence is incorrectly split as in the above example, the first split sentence is concatenated to `text`, then this
                        # condition succeeds and its `continue` statement is executed, then we go to the next iteration of the for-loop where the second split sentence is
                        # contatenated to `text` so that the two incorrectly split sentences are concatenated back together without each components being processed. Then this whole 
                        # sentence will now correctly be processed. 
                        text += sentence + ' '
                        if any([entity['boundaries'][0] < len(text) + prev_len < entity['boundaries'][1] for entity in article['entities']]):
                            continue

                        # Check that the end boundary of the entity is between the previous sentence and the current sentence. Note we don't have to also check that the start
                        # boundary is also between the prev and cur sentence because this is implicitly handled in the above condition. 
                        # Then sort the entities by their start boundary.
                        entities = sorted([entity for entity in article['entities'] if prev_len < entity['boundaries'][1] <= len(text)+prev_len], key=lambda tup: tup['boundaries'][0])
                        # Construct the whole 'linearized triplets' string for all the relations/triplets in this sentence. 
                        # We do this by first grouping all the triplets of this sentence by the subject/head entity. 
                        # Then we construct the 'linearized triplets' string for each head entity.
                        # Then we concatenating these 'linearized triplets' strings of all head entities into a single string, getting us the 
                        # entire 'linearized triplets' of this sentence.
                        # Start with the accumulator `decoder_output` which starts out as the "<triplet> " string.
                        decoder_output = '<triplet> '
                        #
                        # For each entity, get all the triplets that have the entity as the subject/head, and the object/tail of the triplet is also within the current sentence.
                        # Then construct the 'linearized triplets' string for this entity by doing this:
                        # 1. Initialize accumulator string "entity <subj>". 
                        # 2. Then for each triplet in the triplets of this entity, concatenate " tail <obj> rel-type <subj> " to the accumulator string. 
                        # 3. Then remove the last " <subj> ", and add " <triplet> " to the end of the accumulator string, for the next entity and its set of triplets. 
                        for int_ent, entity in enumerate(entities):
                            # For all triplets in the article of this sentence, get all the triplets where the subject is the current entity and the object is within the current sentence.
                            # Then sort these triplets by the start boundary of the object. 
                            triplets = sorted(
                                [
                                    triplet for triplet in article['triples'] 
                                    if (triplet['subject'] == entity and 
                                        prev_len < triplet['subject']['boundaries'][1] <= len(text) + prev_len and 
                                        prev_len < triplet['object']['boundaries'][1] <= len(text)+ prev_len and 
                                        triplet['predicate']['surfaceform'] in relations)
                                ], 
                                key=lambda tup: tup['object']['boundaries'][0]
                            )
                            if len(triplets) == 0:
                                continue
                            decoder_output += entity['surfaceform'] + ' <subj> '
                            # Now iteratively construct the 'linearized triplets' string for all triplets with this entity as the subject.
                            for triplet in triplets:
                                decoder_output += triplet['object']['surfaceform'] + ' <obj> '  + triplet['predicate']['surfaceform'] + ' <subj> '
                            # Remove the last " <subj> " 
                            decoder_output = decoder_output[:-len(' <subj> ')]
                            # Add the " <triplet> " string to the end of the accumulator string to set up for the 
                            # processing of the next set of triplets of the next head entity.  
                            decoder_output += ' <triplet> '
                        # Remove the last " <triplet> " from the accumulator string.
                        decoder_output = decoder_output[:-len(' <triplet> ')]
                        count += 1
                        prev_len += len(text)

                        if len(decoder_output) == 0:
                            text = ''
                            continue

                        text = re.sub('([\[\].,!?()])', r' \1 ', text.replace('()', ''))
                        text = re.sub('\s{2,}', ' ', text)

                        # Finally, yield the article id, and the text (which is again, a single sentence), title, and triplets for this sentence.
                        yield article['uri'] + '-' + str(count), {
                            "title": article['title'],
                            "context": text,
                            "id": article['uri'] + '-' + str(count),
                            "triplets": decoder_output,
                        }
                        text = ''
