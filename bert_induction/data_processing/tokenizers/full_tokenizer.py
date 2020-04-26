# coding=utf-8
import collections

from data_processing.tokenizers import BaseTokenizer
from model.bert.tokenization import FullTokenizer as BertFullTokenizer
import tensorflow as tf


class FullTokenizer(BaseTokenizer):
    def __init__(self, vocab_file):
        self.full_tokenizer = BertFullTokenizer(vocab_file, True)

    def convert_to_vector(self, sentence, max_len):
        tokens = ["[CLS]"]
        tokens.extend(self.full_tokenizer.tokenize(sentence))
        if len(tokens) > max_len - 1:
            tokens = tokens[0:(max_len - 2)]
        tokens.append("[SEP]")
        ids = self.full_tokenizer.convert_ids_to_tokens(tokens)
        mask = [1] * len(ids)
        mask.extend([0] * (max_len - len(mask)))
        ids.extend([0] * (max_len - len(ids)))
        assert len(ids) == max_len
        assert len(mask) == max_len
        return ids, mask
