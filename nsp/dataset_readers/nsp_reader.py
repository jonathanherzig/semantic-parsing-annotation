import logging
from typing import List, Dict
import numpy as np
from overrides import overrides
import json

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, ArrayField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.tokenizers.token import Token
from allennlp.data.fields.field import Field
import torch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NamespaceSwappingField(Field[torch.Tensor]):
    """
    A ``NamespaceSwappingField`` is used to map tokens in one namespace to tokens in another namespace.
    It is used by seq2seq models with a copy mechanism that copies tokens from the source
    sentence into the target sentence.
    Parameters
    ----------
    source_tokens : ``List[Token]``
        The tokens from the source sentence.
    target_namespace : ``str``
        The namespace that the tokens from the source sentence will be mapped to.
    """

    def __init__(self,
                 source_tokens: List[Token],
                 target_namespace: str) -> None:
        self._source_tokens = source_tokens
        self._target_namespace = target_namespace
        self._mapping_array: List[int] = None

    @overrides
    def index(self, vocab: Vocabulary):
        self._mapping_array = [vocab.get_token_index(x.text, self._target_namespace)
                               for x in self._source_tokens]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": len(self._source_tokens)}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_length = padding_lengths["num_tokens"]
        padded_tokens = pad_sequence_to_length(self._mapping_array, desired_length)
        tensor = torch.LongTensor(padded_tokens)
        return tensor

    @overrides
    def empty_field(self) -> 'NamespaceSwappingField':
        return NamespaceSwappingField([], self._target_namespace)


@DatasetReader.register("nsp_reader")
class CopyNetDatasetReader(DatasetReader):
    """
    Read a json file containing paired sequences, and create a dataset suitable for a
    ``CopyNet`` model, or any model with a matching API.
    Parameters
    ----------
    target_namespace : ``str``, required
        The vocab namespace for the targets. This needs to be passed to the dataset reader
        in order to construct the NamespaceSwappingField.
    source_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the input sequences into words or other kinds of tokens. Defaults
        to ``WordTokenizer()``.
    target_tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the output sequences (during training) into words or other kinds
        of tokens. Defaults to ``source_tokenizer``.
    source_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input (source side) token representations. Defaults to
        ``{"tokens": SingleIdTokenIndexer()}``.
    target_token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define output (target side) token representations. Defaults to
        ``source_token_indexers``.
    Notes
    -----
    By ``source_length`` we are referring to the number of tokens in the source
    sentence including the ``START_SYMBOL`` and ``END_SYMBOL``, while
    ``trimmed_source_length`` refers to the number of tokens in the source sentence
    *excluding* the ``START_SYMBOL`` and ``END_SYMBOL``, i.e.
    ``trimmed_source_length = source_length - 2``.
    On the other hand, ``target_length`` is the number of tokens in the target sentence
    *including* the ``START_SYMBOL`` and ``END_SYMBOL``.
    In the context where there is a ``batch_size`` dimension, the above refer
    to the maximum of their individual values across the batch.
    In regards to the fields in an ``Instance`` produced by this dataset reader,
    ``source_token_ids`` and ``target_token_ids`` are primarily used during training
    to determine whether a target token is copied from a source token (or multiple matching
    source tokens), while ``source_to_target`` is primarily used during prediction
    to combine the copy scores of source tokens with the generation scores for matching
    tokens in the target namespace.
    """

    def __init__(self,
                 target_namespace: str,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._target_namespace = target_namespace
        self._source_tokenizer = source_tokenizer or WordTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        if "tokens" not in self._source_token_indexers or \
                not isinstance(self._source_token_indexers["tokens"], SingleIdTokenIndexer):
            raise ConfigurationError("CopyNetDatasetReader expects 'source_token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        self._target_token_indexers: Dict[str, TokenIndexer] = {
                "tokens": SingleIdTokenIndexer(namespace=self._target_namespace)
        }

    @overrides
    def _read(self, file_path):
        # get the split for metadata
        for split_opt in ['train', 'dev', 'test']:
            if split_opt in file_path:
                split = split_opt
                break
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, line in enumerate(data_file):
                line = line.strip("\n")
                line = json.loads(line)
                if not line:
                    continue
                source_sequence, target_sequence, variables = line["nl"], line["lf"], line["variables"]
                if not source_sequence:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence, variables, line_num, split)

    @staticmethod
    def _tokens_to_ids(tokens: List[Token]) -> List[int]:
        ids: Dict[str, int] = {}
        out: List[int] = []
        for token in tokens:
            out.append(ids.setdefault(token.text.lower(), len(ids)))
        return out

    @overrides
    def text_to_instance(self, source_string: str, target_string: str = None, variables: dict = None,
                         index: int = None, split: str = None) -> Instance:  # type: ignore
        """
        Turn raw source string and target string into an ``Instance``.
        Parameters
        ----------
        source_string : ``str``, required
        target_string : ``str``, optional (default = None)
        variables : The entities that appear in the example
        index : the index of the example in the dataset
        Returns
        split: The split (train/dev/test) the example is part of
        -------
        Instance
            See the above for a description of the fields that the instance will contain.
        """
        # pylint: disable=arguments-differ
        tokenized_source = [Token(token) for token in source_string.split(" ")]
        tokenized_source.insert(0, Token(START_SYMBOL))
        tokenized_source.append(Token(END_SYMBOL))
        source_field = TextField(tokenized_source, self._source_token_indexers)

        # For each token in the source sentence, we keep track of the matching token
        # in the target sentence (which will be the OOV symbol if there is no match).
        source_to_target_field = NamespaceSwappingField(tokenized_source[1:-1], self._target_namespace)

        meta_fields = {"source_tokens": [x.text for x in tokenized_source[1:-1]]}
        meta_fields["variables"] = variables
        meta_fields["index"] = index
        meta_fields["split"] = split
        fields_dict = {
                "source_tokens": source_field,
                "source_to_target": source_to_target_field,
        }

        if target_string is not None:
            tokenized_target = [Token(token) for token in target_string.split(" ")]
            tokenized_target.insert(0, Token(START_SYMBOL))
            tokenized_target.append(Token(END_SYMBOL))
            target_field = TextField(tokenized_target, self._target_token_indexers)

            fields_dict["target_tokens"] = target_field
            meta_fields["target_tokens"] = [y.text for y in tokenized_target[1:-1]]
            source_and_target_token_ids = self._tokens_to_ids(tokenized_source[1:-1] +
                                                              tokenized_target)
            source_token_ids = source_and_target_token_ids[:len(tokenized_source)-2]
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))
            target_token_ids = source_and_target_token_ids[len(tokenized_source)-2:]
            fields_dict["target_token_ids"] = ArrayField(np.array(target_token_ids))
        else:
            source_token_ids = self._tokens_to_ids(tokenized_source[1:-1])
            fields_dict["source_token_ids"] = ArrayField(np.array(source_token_ids))

        fields_dict["metadata"] = MetadataField(meta_fields)

        return Instance(fields_dict)
