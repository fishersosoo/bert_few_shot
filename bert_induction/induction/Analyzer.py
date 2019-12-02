# coding=utf-8
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score,f1_score

if __name__ == '__main__':
    query=pd.read_csv(r"Z:\bert_few_shot\data\output\predict\query_set.csv",encoding="utf-8")
    result=pd.read_csv(r"Z:\bert_few_shot\data\result\test_results.csv",encoding="utf-8")
    query["prediction"] = result["prediction"]
    for name,group in query.groupby(["cat","label","class_index"]):
        y_true = query["class_index"]==name[2]
        y_pred = query["prediction"]==name[2]
        f1=f1_score(y_true,y_pred)
        acc=np.sum(group["prediction"]==group["class_index"])/len(group)
        print("{name}:{acc:2f}  {f1}".format(name=name,acc=acc,f1=f1))
    acc=np.sum(query["class_index"] == result["prediction"])/len(query)
    y_true=query["class_index"]
    y_pred=result["prediction"]
    kappa_value = cohen_kappa_score(y_true, y_pred)
    print("random acc: {acc}".format(acc=1/5))
    print("acc: {acc:2f}".format(acc=acc))
    print("kappa: {kappa:2f} ".format(kappa=kappa_value))

    query = pd.read_csv(r"Z:\bert_few_shot\data\output\self_support\predict\query_set.csv", encoding="utf-8")
    result = pd.read_csv(r"Z:\bert_few_shot\data\output\self_support\result\test_results.csv", encoding="utf-8")
    query["prediction"] = result['0']
    for name,group in query.groupby(["cat","label"]):
        mean=np.mean(group["prediction"])
        print("{name}:{mean}±{std}".format(name=name,mean=mean,std=np.std(group["prediction"])))

