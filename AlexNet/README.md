# AlexNet论文的解读

**论文地址 TODO**

## 个人总结
该文章提出了关于网络改善性能，减少训练时间一些技巧(Relu、多块GPU训练、LRN、重叠池化层、dropOut等)，在其paper中的结果中可以看到，确实对于训练过程以及性能有一定的帮助

## Abstract
在LSVRC_2010的测试集中，取得了Top-1的错误率是37.5%，Top-5的错误率是17%(这里解释下什么是Top-1:就是指第一种类别上的精确度或者错误率，这里指的错误率，Top-5指的是前五种类别的总错误率)。该网络有60M 的参数量650000的神经元数量.
5个卷积层，某些卷积层后也跟着池化层，随后还有三层全连接，最后softmax删除分类1000.同时为了更快的训练，使用了非饱和神经元（non-saturating），以及有效的GPU训练方式。
同时为了减少过拟合，使用了dropOut.最终在ILSVRC_2012中，取得了top-5 错误率15.3%.

ps： 这里不知道是不是首次使用了DropOut，但是在之前的看的论文当中，并没有专门体积DropOut的使用。

## 1 Introduction
**略过前面，看后面段落重点:**

最后，网络的大小主要受到当前GPU上可用内存量的限制 我们愿意容忍的训练时间。我们的网络需要5分钟 在两个GTX 580 3GB GPU上训练6天。我们所有的实验都表明我们的结果 只需等待更快的GPU和更大的数据集可用，就可以提高性能。

(In the end, the network’s size is limited mainly by the amount of memory available on current GPUs
and by the amount of training time that we are willing to tolerate. Our network takes between five
and six days to train on two GTX 580 3GB GPUs. All of our experiments suggest that our results
can be improved simply by waiting for faster GPUs and bigger datasets to become available.)

典型的多卡训练，但是GTX580显卡不太行了，3GB也有点小。本地是1660Ti 6GB虽然同样渣，但是完全能达到作者训练的效果

## 2 The Dataset
**略**

## 3 The Architecture

![img_1](https://user-images.githubusercontent.com/28073374/134807475-6c87e656-8e39-4827-8ebf-2e432549fbb9.png)

**关于此结构的一些解释：**

dense：即全连接层

### 3.1 ReLU Nonlinearity

这里要好好看看了，这里提出了Relu的使用。

之前网络中常用的激活函数都是tanh（ tanh(x)）或者sigmoid（f(x) = (1 + e −x)−1）

(The standard way to model a neuron’s output f as
a function of its input x is with f(x) = tanh(x)
or f(x) = (1 + e −x)−1)

但是上述的激活函数即饱和神经元，比非饱和神经元训练更慢(Relu)

(In terms of training time
with gradient descent, these saturating nonlinearities
are much slower than the non-saturating nonlinearity
f(x) = max(0, x).)

一些具体的差异，如图。关于为何使用Relu,主要目的就是因为更快吧

另外关于non-saturating neurons的解释：应该就是output值并不在某个特定的区间，查了一些网上的资料，可能这种说法比较靠谱

![img_2](https://user-images.githubusercontent.com/28073374/134807478-22ca97c9-d6b9-45d3-94a4-06d066045b9b.png)

### 3.2 Training on Multiple GPUs
关于多GPU的训练，略

### 3.3 Local Response Normalization

局部响应标准化，LRN层，一个Normalization 操作，这篇具体讲的啥呢。先看公式，后来看网上很多人说这个被各种bn替代了。

![img_3](https://user-images.githubusercontent.com/28073374/134807482-9902f62b-b6cf-48b1-932b-a4f8c05842b8.png)


a^i x,y:表示第 i 片特征图在位置（x,y）运用激活函数 ReLU 后的输出
N:是特征图的总数
n:是同一位置上临近的 feature map 的数目
k,β,α,n:都是参数

### 3.4 Overlapping Pooling

重叠池化层，

池的大小为z*z,如果s(步长)小于z的话，则作为重叠池化层

(If we set s < z, we obtain overlapping pooling) 

再看看重叠池化层的作用，s=2 ，z=3 比较s=z=2.前者最后的结果好，也一定程度上对解决过度拟合有帮助

(This is what we use throughout our
network, with s = 2 and z = 3. This scheme reduces the top-1 and top-5 error rates by 0.4% and
0.3%, respectively, as compared with the non-overlapping scheme s = 2, z = 2, which produces
output of equivalent dimensions. We generally observe during training that models with overlapping
pooling find it slightly more difficult to overfit.)

### 3.5 Overall Architecture

回顾前面的结构图：

8层网络，五层卷积+三层全连接，最后一层的输出通过softmax直接输出1000标签

(Now we are ready to describe the overall architecture of our CNN. As depicted in Figure 2, the net
contains eight layers with weights; the first five are convolutional and the remaining three are fully connected. The output of the last fully-connected layer is fed to a 1000-way softmax which produces
a distribution over the 1000 class labels. Our network maximizes the multinomial logistic regression
objective, which is equivalent to maximizing the average across training cases of the log-probability
of the correct label under the prediction distribution.)

再看这一句话，Relu接在了每个卷积和全连接之后

(The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer)

由于是双路GPU进行训练，在本地是单路GPU，所以实际训练，关于通道数啥的 都应该*2使用

## 4 Reducing Overfitting

接下来讲两个主要的方式来防止过拟合

### 4.1 Data Augmentation

**第一种是通过图像的平移和水平反射**

（The first form of data augmentation consists of generating image translations and horizontal reflections）

**第二种是改变图像中RGB通道的强度训练图像**

（The second form of data augmentation consists of altering the intensities of the RGB channels in
training images）

**取得的效果:**

This scheme reduces the top-1 error rate by over 1%.

## 4.2 Dropout
Drop out p设置为0.5

## 5 Details of learning

**接下来时讲关于学习的细节**

![img_4](https://user-images.githubusercontent.com/28073374/134807506-c61005c0-38a3-41bd-a19e-e9d5226a62bf.png)

一个逐渐递减的学习

where i is the iteration index, v is the momentum variable,  is the learning rate, and D ∂L ∂w wi E Di is the average over the ith batch Di of the derivative of the objective with respect to w, evaluated at wi . 

## 6 Results
略

## 7 Discussion
略

