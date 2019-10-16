import os

import pandas as pd

from bert import tokenization
from bert.run_classifier import DataProcessor, InputExample


class ShoppingDataProcessor(DataProcessor):
    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        return self._create_examples(df, "train")

    def get_dev_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, "dev.csv"))
        return self._create_examples(df, "dev")

    def get_test_examples(self, data_dir):
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        return self._create_examples(df, "test")

    def _create_examples(self, df, set_type):
        examples = []
        for i, row in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(df["text"])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(df["class"])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples
