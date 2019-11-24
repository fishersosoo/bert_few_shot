# coding=utf-8
import os

import numpy as np
import jieba
import six


class Tokenizer():
    def __init__(self, path, dict_path=None,mode='text'):
        self.OOV = set()
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        if mode=="text":
            with open(path, encoding='utf-8', errors='ignore') as f:
                first_line = True
                for line in f:
                    if first_line:
                        first_line = False
                        dim = int(line.rstrip().split()[1])
                        continue
                    lines_num += 1
                    tokens = line.rstrip().split(' ')
                    vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
                    iw.append(tokens[0])
        for i, w in enumerate(iw):
            wi[w] = i
        print("Vectors loaded ")
        self.vectors = vectors
        self.wi = wi
        self.iw = iw
        self.dim = dim
        if dict_path is not None and not os.path.exists(dict_path):
            with open(dict_path, 'w', encoding="utf-8") as dict_file:
                dict_file.write('\n'.join(self.iw))
            jieba.load_userdict(dict_path)

    @classmethod
    def convert_to_unicode(cls, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")

    def convert_to_vector(self, sentence, max_len):
        words = [one for one in jieba.cut(self.convert_to_unicode(sentence))][:max_len]
        vector = []
        for word in words:
            vector.extend(self.word_to_vector(word))
        # trunc
        vector = vector[:max_len]
        # padding
        vector.extend([[0.0] * self.dim] * (max_len - len(vector)))
        return vector

    def word_to_vector(self, word):
        vectors = []
        vector = self.vectors.get(word, None)
        if vector is None:
            for char in word:
                char_vector = self.vectors.get(char, None)
                if char_vector is None:
                    self.OOV.add(char)
                    char_vector = [0.] * self.dim
                vectors.append(char_vector)
        else:
            vectors.append(vector)
        return vectors


if __name__ == '__main__':

    support_input_text = np.load(os.path.join("/home/bert_few_shot/data/output/train", "support_text.npy"),
                                 allow_pickle=True)
    support_input_text = support_input_text.reshape(-1)
    size = support_input_text.shape[0]
    tokenizer = Tokenizer("/home/bert_few_shot/models/vector/merge_sgns_bigram_char300.txt",
                          "/home/bert_few_shot/models/vector/user_dict.txt")
    for i, text in enumerate(support_input_text):
        if i % (size / 10) == 0:
            print("{done}%...".format(done=int(i / size * 100)))
        tokenizer.convert_to_vector(text, 64)
    with open("/home/bert_few_shot/data/output/train/OOV.txt", "w", encoding="utf-8") as oov_file:
        oov_file.write('\n'.join((one for one in tokenizer.OOV)))
        oov_file.flush()
