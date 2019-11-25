# few shot

## Changelists

### 2019-11-25

- 复现原文实现，使用单层的双向LSTM使用attention机制进行编码
- k=5, c=1, hidden_size=128, attention_size=64, dropout_prob=0.1, max_length=32, batch_size=32, epochs=32

测试效果：

('书籍', 0):0.79
('水果', 1):0.73
('洗发水', 0):0.27
('热水器', 0):0.61
('热水器', 1):0.63

random acc: 0.2
acc: 0.61
kappa: 0.51

