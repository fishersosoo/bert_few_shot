# coding=utf-8
import os

import numpy as np
import jieba

from data_processing.tokenizers import BaseTokenizer


class SegmentFixedTokenizer(BaseTokenizer):
    """
    用 jieba 分词，每个词去找预训练好的固定词向量
    Examples:
        sentences=["句子1","句子2"]
        tokenizer = SegmentFixedTokenizer("/home/bert_few_shot/models/vector/merge_sgns_bigram_char300.txt",
                                          "/home/bert_few_shot/models/vector/user_dict.txt")
        for i, text in enumerate(sentences):
            tokenizer.convert_to_vector(sentences, 64)

        # 输出 OOV 字符
        with open("/home/bert_few_shot/data/output/train/OOV.txt", "w", encoding="utf-8") as oov_file:
            oov_file.write('\n'.join((one for one in tokenizer.OOV)))
            oov_file.flush()
    """

    def __init__(self, path, dict_path=None):
        """

        Args:
            path: 预训练词向量路径
            dict_path: 分词字典路径
            mode:
        """
        self.OOV = set()
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
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

    def convert_to_vector(self, sentence, max_len):
        words = [one for one in jieba.cut(self.convert_to_unicode(sentence))][:max_len]
        vector = []
        for word in words:
            vector.extend(self._word_to_vector(word))
        # trunc
        vector = vector[:max_len]
        # padding
        vector.extend([[0.0] * self.dim] * (max_len - len(vector)))
        return vector

    def _word_to_vector(self, word):
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

