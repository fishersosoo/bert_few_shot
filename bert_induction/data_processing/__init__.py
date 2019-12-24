# encoding=utf-8
from data_processing.data_set.online_shopping_data import OnlineShoppingData
from data_processing.tokenizers.segment_fixed_tokenizer import SegmentFixedTokenizer

all_data_set = {
    "online_shopping": OnlineShoppingData
}

all_tokenizer = {
    "segment_fixed_tokenizer": SegmentFixedTokenizer
}


def get_dataset_cls(name):
    name = name.lower()
    return all_data_set.get(name, None)


def get_tokenizer_cls(name):
    name = name.lower()
    return all_tokenizer.get(name, None)
