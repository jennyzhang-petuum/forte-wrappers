# Copyright 2019 The Forte Authors. All Rights Reserved.
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
Unit tests for HFSentimentAnalysis processor.
"""
import unittest
from typing import Dict
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import StringReader
from forte.nltk import NLTKSentenceSegmenter
from forte.huggingface.sentiment_analysis import HFSentimentAnalysis
from ft.onto.base_ontology import Document, Sentence

# from tests.helpers.test_utils import get_top_scores_label


def get_top_scores_label(curr_dict: Dict):
    top1 = sorted(curr_dict.keys(), key=lambda x: curr_dict[x], reverse=True)[0]
    return top1


class TestHFSentimentAnalysis(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack](enforce_consistency=True)
        self.nlp.set_reader(StringReader())
        conf = {
            "entry_type": "ft.onto.base_ontology.Document",
        }
        self.nlp.add(HFSentimentAnalysis(), config=conf)
        self.nlp.add(NLTKSentenceSegmenter())
        self.nlp.add(HFSentimentAnalysis())
        self.nlp.initialize()

    def test_huggingface_sentiment_analysis_processor(self):
        sentences = [
            "I love using transformers.",
            "But I'm worried about the implementation challenges.",
            "It's amazing how Forte makes transformers easy to use.",
        ]
        document = " ".join(sentences)
        pack = self.nlp.process(document)
        # Document
        expected_scores_d = [
            {
                "sadness": 0.004,
                "joy": 0.073,
                "love": 0.001,
                "anger": 0.005,
                "fear": 0.211,
                "surprise": 0.706,
            }
        ]
        expected_tops_d = [get_top_scores_label(x) for x in expected_scores_d]
        for idx, sentence in enumerate(pack.get(Document)):
            self.assertEqual(
                get_top_scores_label(
                    sentence.classifications["sentiment"].classification_result
                ),
                expected_tops_d[idx],
            )
        # Sentence
        expected_scores_s = [
            {
                "sadness": 0.008,
                "joy": 0.833,
                "love": 0.014,
                "anger": 0.136,
                "fear": 0.008,
                "surprise": 0.002,
            },
            {
                "sadness": 0.003,
                "joy": 0.003,
                "love": 0.001,
                "anger": 0.033,
                "fear": 0.960,
                "surprise": 0.000,
            },
            {
                "sadness": 0.004,
                "joy": 0.558,
                "love": 0.002,
                "anger": 0.004,
                "fear": 0.007,
                "surprise": 0.425,
            },
        ]
        expected_tops_s = [get_top_scores_label(x) for x in expected_scores_s]
        for idx, sentence in enumerate(pack.get(Sentence)):
            self.assertEqual(
                get_top_scores_label(
                    sentence.classifications["sentiment"].classification_result
                ),
                expected_tops_s[idx],
            )


if __name__ == "__main__":
    unittest.main()
