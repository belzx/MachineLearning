## Yolo论文的代码实现
### 运行环境 
1：python 3.9

2：下载VOC2007

    训练数据：解压到Yolo/data/train/Voc2007

    测试数据：解压到Yolo/data/test/Voc2007

**参考**

![](.\local_data\1.png)

3:安装依赖: pip install -r requirements.txt
### Yolo-v1
**完成度80%**
**已经完成**
1：训练模型、预测代码已经完成

**未完成**
1：GPU训练改造
2：日志输出
3：模型还需要调试

### Yolo-9000
TODO
### Yolo-3
TODO
### Yolo-5
TODO

### AlexNet
**完成度80%**
**已经完成**
1：训练模型、测试、预测代码已经完成

### MnistNet
Mnist数据集识别CNN网络，训练20个循环后，测试正确率百分97
**完成度100%**

### LeNet5
熟悉模型，看论文

### ResNet
TODO

### GoogleNet
见GoogLeNet模块下README

### VGG
TODO

### R-CNN
TODO


### other
How Does Batch Normalization Help Optimization?
https://arxiv.org/abs/1805.11604

如何设计CNN网络的文章
A practical theory for designing very deep convolutional neural networks

```
一些在使用中记录的疑点
sigmoid() 与 softmax()的区别
sigmoid:多标签问题
softmax:单标签问题，互斥输出
https://zhuanlan.zhihu.com/p/69771964

log_softmax:结果存在负数
```


```
参数量计算：
输入[64,3,28,28]
1:卷积层：输入通道:3 输出128，卷积3*3 则参数量：64*3*3*3*128 = 157184 输出[64,128,28,28]
2:拉平:output->[64,100352] 
3:全连接层:输入:[64,100352] 输出[64,10]，全连接层nn.Liner(100352,10)  y = xA^T + b 线性代数里面[1xn]*[n*m] + [1*m]= [1*m]+b 
n = 100352, m = 10 ,m = 10
则参数量为:64*(100352*10) = 64225280

全连接层的参数量远远大于卷积层参数量

卷积核设计
参数设计：
1*1：用于调整通道数量，升维或者降维（用于设计瓶颈式结构） 典型应用 ResNet中

3*3 5*5：使用更小的卷积比较有利
3*3卷积参数量更小，网络层次增加，表达能力更好

膨胀卷积：不增加实际计算量，但是有更大的感受野
一般为3*3 步长为1.大小一般为奇数，越大朝向

全连接层的作用:
全连接层和卷积层没有太大区别
卷积层像是对于计算量以及准确度的一种妥协

模型设计:
卷积-激活-池化
全连接-激活
softmax sigmod


一些在使用中记录的疑点
sigmoid() 与 softmax()的区别
sigmoid:多标签问题
softmax:单标签问题，互斥输出
https://zhuanlan.zhihu.com/p/69771964

log_softmax:结果存在负数

在训练mnist时不同的优化算法
SGD：(Stochatic Gradient decent )
1：随机采样 2：计算平均损失梯度 3：计算衰减学习率 4：修改权值
BGD: (Batch Gradient Descent)
基于所有样本进行梯度下降（一次epoch），最大能全局最优，但是速度很慢


各个参数含义:

损失函数:
cross_entropy:交叉熵损失函数
本质上是一种对数似然函数，多用于二分类或者分类中，用这个loos前面还不需要加Softmax层
https://zhuanlan.zhihu.com/p/58883095
mse_loss: (Mean Squared Error)
https://zhuanlan.zhihu.com/p/346935187
nll_loss:
适合最后一层是log_sooftmax
https://www.cnblogs.com/ranjiewen/p/10059490.html

参数含义:
```

