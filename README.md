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


### other
```
一些在使用中记录的疑点
sigmoid() 与 softmax()的区别
sigmoid:多标签问题
softmax:单标签问题，互斥输出
https://zhuanlan.zhihu.com/p/69771964

log_softmax:结果存在负数

在训练mnist时不同的优化器
SGD
ADM
各个参数含义:

损失函数:
cross_entropy:
mse_loss:
nll_loss:
参数含义:

```

