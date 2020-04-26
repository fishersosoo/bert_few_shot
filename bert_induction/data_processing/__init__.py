# encoding=utf-8
from data_processing.data_set.online_shopping_data import OnlineShoppingData
from data_processing.record_codec import EmbeddingsRecordCodec
from data_processing.tokenizers.full_tokenizer import FullTokenizer
from data_processing.tokenizers.segment_fixed_tokenizer import SegmentFixedTokenizer

all_data_set = {
    "online_shopping": OnlineShoppingData
}

all_tokenizer = {
    "segment_fixed_tokenizer": SegmentFixedTokenizer,
    "full_tokenizer":FullTokenizer
}
all_record_codec = {
    "embeddings_record_codec": EmbeddingsRecordCodec
}


def get_record_codec(name):
    name = name.lower()
    return all_record_codec.get(name, None)


def get_dataset_cls(name):
    name = name.lower()
    return all_data_set.get(name, None)


def get_tokenizer_cls(name):
    name = name.lower()
    return all_tokenizer.get(name, None)
