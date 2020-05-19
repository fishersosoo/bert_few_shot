# encoding=utf-8
from data_processing.data_set.online_shopping_data import OnlineShoppingData
from data_processing.tokenizers.full_tokenizer import FullTokenizer
from data_processing.tokenizers.segment_fixed_tokenizer import SegmentFixedTokenizer

all_data_set = {
    "online_shopping": OnlineShoppingData
}

all_tokenizer = {
    "segment_fixed_tokenizer": SegmentFixedTokenizer,
    "full_tokenizer": FullTokenizer
}


def get_dataset_cls(name):
    name = name.lower()
    return all_data_set.get(name, None)


def get_tokenizer(meta_dict):
    cls = all_tokenizer.get(meta_dict["tokenizer"]["name"])
    return cls(meta_dict)
