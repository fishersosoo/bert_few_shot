## docker相关命令

启动容器

端口使用

| host端口 | 容器端口 | 用途        |
| -------- | -------- | ----------- |
| 6180     | 80       | tensorboard |
| 6122     | 22       | ssh         |
| 6181     | 81       | notebook    |
| 6194     | 1194     | openvpn     |



```bash
nvidia-docker run -itd --runtime=nvidia -p 6180:80 -p 6122:22 -p 6181:81 -p 6194:1194 --name="liangzhanning_tf" --cap-add NET_ADMIN -v /sdb/data/liangzhanning/bert_few_shot:/home/bert_few_shot liangzhanning_tensorflow /bin/bash -c "/usr/sbin/sshd -D;screen -wipe;/bin/bash"
```

```bash
docker ps
```



容器重命名

```bash
docker rename 原容器名  liangzhanning_tf
```

进入容器

```bash
nvidia-docker exec -it 容器id /bin/bash
```

修改语言

```bash
export LC_ALL=zh_CN.UTF-8
export LANG=zh_CN.UTF-8
export LANGUAGE=zh_CN.UTF-8
```

```bash
screen -S jupyter
```

```bash
jupyter notebook --port=81 --ip='0.0.0.0' --notebook-dir='/home/bert_few_shot' --no-browser --allow-root

```

```bash
docker commit liangzhanning_tf  liangzhanning_tensorflow
```





## bert命令

```bash
cd /home/bert_few_shot/bert_induction
```



bert模型文本分类预测

```bash
python bert/run_classifier.py --task_name=shopping \
--do_predict=true \
--data_dir=/home/bert_few_shot/data  \
--vocab_file=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=128 \
--output_dir=/home/bert_few_shot/data/test >/home/bert_few_shot/log 2>&1
```



bert模型文本分类训练

```bash
CUDA_VISIBLE_DEVICES=3 python bert/run_classifier.py --task_name=shopping \
--do_train=true \
--data_dir=/home/bert_few_shot/data \
--vocab_file=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=/home/bert_few_shot/models/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=64 \
--save_checkpoints_steps=30000 \
--train_batch_size=32  \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--use_tpu=false  \
--output_dir=/home/bert_few_shot/models/test_fine  >/home/bert_few_shot/log 2>&1
```

tensorboard 启动

```ba
tensorboard --host=0.0.0.0 --port=80 --logdir=/home/bert_few_shot/models/trained/induction
```

