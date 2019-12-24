# coding=utf-8
from abc import abstractmethod

import six


class BaseTokenizer(object):

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

    @abstractmethod
    def convert_to_vector(self, sentence, max_len):
        """

        Args:
            sentence:
            max_len:

        Returns:
            float array [max_len, dim]. dim 是向量维度
        """
        raise NotImplementedError()

