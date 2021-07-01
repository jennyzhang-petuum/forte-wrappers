# Copyright 2021 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Wrapper of the Sentiment Analysis models on HuggingFace platform
"""
from typing import Dict, Set
import importlib

from transformers import pipeline
from ft.onto.base_ontology import Classification
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor

__all__ = [
    "HFSentimentAnalysis",
]


class HFSentimentAnalysis(PackProcessor):
    r"""Wrapper of the models on HuggingFace platform with pipeline tag of
    `text-classification`.
    https://huggingface.co/models?pipeline_tag=text-classification
    This wrapper could take any model name on HuggingFace platform with pipeline
    tag of `sentiment-analysis` in configs to make prediction on the user
    specified entry type in the input pack and the prediction result goes to the
    user specified attribute name of that entry type in the output pack.

    """

    def __init__(self):
        super().__init__()
        self.classifier = None

    def set_up(self):
        device_num = self.configs["cuda_device"]
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.configs.model_name,
            framework="pt",
            device=device_num,
            return_all_scores=True,
        )

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.set_up()

    def _process(self, input_pack: DataPack):
        path_str, module_str = self.configs.entry_type.rsplit(".", 1)

        mod = importlib.import_module(path_str)
        entry = getattr(mod, module_str)
        for entry_specified in input_pack.get(entry_type=entry):
            result = self.classifier(entry_specified.text)
            curr_dict = getattr(entry_specified, self.configs.attribute_name)
            cls_field = Classification(input_pack)
            res_dict = dict()
            for res in result[0]:
                res_dict[res["label"]] = round(res["score"], 4)
            cls_field.classification_result = res_dict
            curr_dict[self.configs.save_to_key] = cls_field
            setattr(entry_specified, self.configs.attribute_name, curr_dict)

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for HFSentimentAnalysis.

        Following are the keys for this dictionary:
            - `entry_type`: defines which entry type in the input pack to make
              prediction on. The default makes prediction on each `Sentence`
              in the input pack.
            - `attribute_name`: defines which attribute of the `entry_type`
              in the input pack to save prediction to. The default
              saves prediction to the `classification` attribute for each
              `Sentence` in the input pack.
            - `model_name`: language model
              The wrapper supports Hugging Face models with pipeline tag of
              `text-classification`.
            - `cuda_device`: Device ordinal for CPU/GPU supports. Setting
              this to -1 will leverage CPU, a positive will run the model
              on the associated CUDA device id.

        Returns: A dictionary with the default config for this processor.
        """
        config = super().default_configs()
        config.update(
            {
                "entry_type": "ft.onto.base_ontology.Sentence",
                "attribute_name": "classifications",
                "save_to_key": "sentiment",
                "model_name": "bhadresh-savani/distilbert-base-uncased-emotion",
                "cuda_device": -1,
            }
        )
        return config

    def expected_types_and_attributes(self):
        r"""Method to add user defined expected type which
        would be checked before running the processor if
        the pipeline is initialized with
        `enforce_consistency=True` or
        :meth:`~forte.pipeline.Pipeline.enforce_consistency` was enabled for
        the pipeline.
        """
        return {self.configs["entry_type"]: set()}

    def record(self, record_meta: Dict[str, Set[str]]):
        r"""Method to add output type record of `ZeroShotClassifier` which is
        user specified entry type with user specified attribute name
        to :attr:`forte.data.data_pack.Meta.record`.

        Args:
            record_meta: the field in the datapack for type record that need to
                fill in for consistency checking.
        """
        record_meta[self.configs.entry_type].add(self.configs.attribute_name)
