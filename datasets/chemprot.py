from __future__ import absolute_import, division, print_function 
import json
import logging
import datasets

_DESCRIPTION = "Chemprot Dataset"

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

class ChemprotConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(ChemprotConfig, self).__init__(**kwargs)
 
class Chemprot(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ChemprotConfig(
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
                    "context": datasets.Value("string"),
                    "triplets": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03889-5",
        )

    def _split_generators(self, dl_manager):
       
        if self.config.data_files:
            downloaded_files = {
                "train": self.config.data_files["train"], 
                "dev": self.config.data_files["dev"], 
                "test": self.config.data_files["test"],
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logging.info("generating examples from = %s", filepath[0])
        with open(filepath[0], 'r') as json_file:
            for id_, line in enumerate(json_file):
                row = json.loads(line)  # Parse each line as a JSON object
                yield str(id_), {
                    "context": row['sentence'],
                    "id": str(id_),
                    "triplets": row['lin_triplets'],
                }
