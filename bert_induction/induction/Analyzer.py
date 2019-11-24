# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

if __name__ == '__main__':
    query=pd.read_csv(r"Z:\bert_few_shot\data\output\predict\query_set.csv",encoding="utf-8")
    result=pd.read_csv(r"Z:\bert_few_shot\data\result\test_results.csv",encoding="utf-8")
    acc=np.sum(query["class_index"] == result["prediction"])/len(query)
    y_true=query["class_index"]
    y_pred=result["prediction"]
    kappa_value = cohen_kappa_score(y_true, y_pred)
    print("random acc: {acc}".format(acc=1/5))
    print("acc: {acc:2f}".format(acc=acc))
    print("kappa: {kappa} ".format(kappa=kappa_value))
