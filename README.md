# transfer-learning
inception-v3    transfer learning  迁移学习

### input-data-v1  用来把划分数据，并制作为numpy的格式
### input-data-v2  用来随机划分数据，并制作为numpy的格式(一次只能划分一种神经束）
（每一个人有8张图片，一共有96个人，一个人的图片要全作为训练集或全作为测试集，所以不能简单随机划分）
### input-data-v3  用来随机划分数据，并制作为numpy的格式（能一次划分所有的28根神经束）


### main_V1.py 把numpy数据训练和测试,可以画出ROC曲线
### main_V2.py 实现了多次随机重复实验，并把多次实验的平均结果自动保存
