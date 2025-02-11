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

# pylint: disable=attribute-defined-outside-init
from typing import Dict, Any

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.top import Query
from forte.processors.base import MultiPackProcessor
from ft.onto.base_ontology import Document

from forte_wrapper.elastic.elastic_indexer import ElasticSearchIndexer

__all__ = [
    "ElasticSearchProcessor"
]


class ElasticSearchProcessor(MultiPackProcessor):
    r"""This processor searches for relevant documents for a query"""

    # pylint: disable=useless-super-delegation
    def __init__(self) -> None:
        super().__init__()

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.index = ElasticSearchIndexer(config=self.configs.index_config)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        """This defines a basic config structure for ElasticSearchProcessor
        Returns:
            A dictionary with the default config for this processor.
            query_pack_name (str): The query pack's name, default is "query".
            index_config (dict): The ElasticSearchIndexer's config.
            field (str): Field name that will be used when creating the new
                datapack.
            response_pack_name_prefix (str): the pack name prefix to be used
                in response datapacks.
            indexed_text_only (bool): defines whether the returned
                value from the field (as specified by the field
                configuration) will be considered as plain text. If True,
                a new data pack will be created and the value will be
                used as the text for the data pack. Otherwise, the returned
                value will be considered as serialized data pack, and the
                returned data pack will be created by deserialization.
                Default is True.
        """
        config = super().default_configs()
        config.update({
            "query_pack_name": "query",
            "index_config": ElasticSearchIndexer.default_configs(),
            "field": "content",
            "response_pack_name_prefix": "passage",
            "indexed_text_only": True
        })
        return config

    def _process(self, input_pack: MultiPack):
        r"""Searches `Elasticsearch` indexer to fetch documents for a query.
        This query should be contained in the input multipack with name
        `self.config.query_pack_name`.

        This method adds new packs to `input_pack` containing the retrieved
        results. Each result is added as a `ft.onto.base_ontology.Document`.

        Args:
             input_pack: A multipack containing query as a pack.
        """
        query_pack = input_pack.get_pack(self.configs.query_pack_name)

        # ElasticSearchQueryCreator adds a Query entry to query pack. We now
        # fetch it as the first element.
        first_query: Query = query_pack.get_single(Query)
        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: until fix: https://github.com/PyCQA/pylint/issues/3507
        if not isinstance(first_query.value, Dict):
            raise ValueError(
                "The query to the elastic indexer need to be a dictionary.")
        results = self.index.search(first_query.value)
        hits = results["hits"]["hits"]

        for idx, hit in enumerate(hits):
            document = hit["_source"]
            first_query.add_result(document["doc_id"], hit["_score"])

            if self.configs.indexed_text_only:
                pack: DataPack = input_pack.add_pack(
                    f"{self.configs.response_pack_name_prefix}_{idx}"
                )
                pack.pack_name = document["doc_id"]

                content = document[self.configs.field]
                pack.set_text(content)

                Document(pack=pack, begin=0, end=len(content))

            else:
                pack = DataPack.deserialize(document['pack_info'])
                input_pack.add_pack_(
                    pack, f"{self.configs.response_pack_name_prefix}_{idx}")
                pack.pack_name = document["doc_id"]
