# coding=utf-8
import json
import tensorflow as tf
import copy
import six


class BaseModelConfig(object):
    @classmethod
    def from_dict(cls, json_obj):
        """

        Args:
            json_obj:

        Returns:

        """
        config = cls()
        for (k, v) in six.iteritems(json_obj):
            config.__dict__[k] = v
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """

        Args:
            json_file: json文件路径

        Returns:

        """
        with tf.gfile.GFile(json_file) as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
