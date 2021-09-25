# GoogleNet论文的解读
**论文地址 https://arxiv.org/abs/1409.4842**

**前面略过，直接第三章开始**

## 3 Motivation and High Level Considerations

一般来说提升网络性能最直接的办法就是增加网络深度和宽度，但是这种简单的解决方式有两个问题:

### 1:巨量的参数会更容易导致过拟合

### 2:对计算资源需求的剧增

这两个问题如何解决呢：

作者认为

根本的解决方法在于:从全连接转为稀疏连接的结构，即使在卷积层内部

怎么去理解这句话呢，我理解是如一个5*5的卷积可以用两层3*3的卷积代替；相较于5*5的卷积连接，3*3的卷积连接是稀疏的.

（The fundamental way of solving both issues would be by ultimately moving from fully connected
to sparsely connected architectures, even inside the convolutions.）

而且由于前人Arora等的开创性工作 ，稀疏的结构也具有更加坚定的理论基础的优势。他们研究的结果宣称：

如果数据集的概率分布是用一个大的且非常稀疏的深度神经网络表现出来的话， 那么最优的网络拓扑可以被一层一层的进行构建：分析上一层激活值的相关联统计数据并对这些具有高度关联性的输出值的神经元进行聚类

(Their main result states that if the probability distribution of
the data-set is representable by a large, very sparse deep neural network, then the optimal network
topology can be constructed layer by layer by analyzing the correlation statistics of the activations
of the last layer and clustering neurons with highly correlated outputs)

尽管数学根据要求非常难以证明以上理论，但是事实—陈述引起了与著名的海扁准则 ---一起产生动作电位的神经元，他们相互连线

(neurons that fire together, wire together)

表明Arora的基础观点是可使用的，即使在实际中缺乏严格的证明条件。

### 再看后面：

Inception架构起初是作为第一作者的一个案例研究： 评估一个复杂的网络拓扑构造算法的假设性输出结果， 该算法试图去逼近一个由Arora暗示的关于视觉网络的稀疏的架构， 并通过稠密的很容易获得的组成成分去覆盖该假设性输出值

（此处覆盖意思是，在该假设性输出值上面接一层dense matrix）。

尽管这是一个具有高度推理性的任务，但是仅仅两次迭代正确的拓扑选择后，我们已经看到了适度的收获。 在进一步的调整学习速率、超参以及改良了的训练方法后，

我们证实了：Inception在localization和object detection的环境中尤其有用。

(The Inception architecture started out as a case study of the first author for assessing the hypothetical
output of a sophisticated network topology construction algorithm that tries to approximate a sparse
structure implied by [2] for vision networks and covering the hypothesized outcome by dense, readily available components. Despite being a highly speculative undertaking, only after two iterations
on the exact choice of topology, we could already see modest gains against the reference architecture based on [12]. After further tuning of learning rate, hyperparameters and improved training
methodology, we established that the resulting Inception architecture was especially useful in the
context of localization and object detection as the base network for [6] and [5]. Interestingly, while
most of the original architectural choices have been questioned and tested thoroughly, they turned
out to be at least locally optimal.)

## 4 Architectural Details
### 主要的思路：
Inception体系结构的主要思想是基于找出最优局部稀疏 卷积视觉网络中的结构可以用现成的方法来近似和覆盖,致密组分

(The main idea of the Inception architecture is based on finding out how an optimal local sparse
structure in a convolutional vision network can be approximated and covered by readily available
dense components.)

Arora 提出了一个层堆叠的结构，其中我们应该分析上一层的相关联的统计数据，并且将这些神经元聚类成具有高度关联性的不同单元群。

（Arora et al. [2] suggests a layer-by layer construction in which one should analyze
the correlation statistics of the last layer and cluster them into groups of units with high correlation）

![2](https://user-images.githubusercontent.com/28073374/134776139-e6510e27-9649-4db6-8cbf-beeff31e37df.png)

图a：使用了1*1 3*3 5*5 同时并行池有效果，又加了一个并行池

图b:由于图a但是会存在问题：的5*5卷积也会带来很多计算的增加，

（One big problem with the above modules, at least in this na¨ıve form, is that even a modest number of
5×5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number
of filters）

所以在3*3 5*5卷积前加了1*1卷积，目的是先降低输入的通道数，从而降低计算量

（That is, 1×1 convolutions are used to compute reductions before the expensive 3×3 and 5×5 convolutions）

通常，一个Inception网络是一个包含了很多以上类型的模块，这些模块一层一层的堆叠， 偶尔步长为2的 max-pooling layers把表格的分辨率减少为原来的一半（即feature map大小减少为原来的1/4大）。
因为技术的原因（训练时内存效率），仅仅只在更高的网络层开始使用Inception模块，而在较低层保持传统的卷积机制，这样的组合是有益的

（it seemed beneficial to start using Inception modules only at higher layers while keeping the lower 
layers in traditional convolutional fashion）

总结：按我的理解就是避免更深，所以在同一级运行多种尺寸的滤波核，让网络更宽。为了避免计算量增加，提前使用1*1滤波先降低维度。

## 5：GoogLeNet
### 具体网络设计

![3](https://user-images.githubusercontent.com/28073374/134776148-bb82c88b-4ad7-400e-be91-9a677961f747.png)

GoogLeNet incarnation of the Inception architecture

### 关于列表解读
depth:depth为2的话则算作两层，0表示不算

-#3x3:reduce表示在3x3卷积操作之前使用了1x1卷积后的输出通道数

3x3:表示3*3输出通道数

params:卷积核的params计算公式，(Cin*Ksize*Ksize*Cout) + Cout 第一个卷积核的参数量应该差不多是9.4k，但是table中的是2.7K,这里有出入不太懂怎么弄的.
参考了网上的说法(7+7)*3*64 ~ 2.7K,应该是计算错误

ops:第一个卷积核的参数量,（Cin*Ksize*Ksize*Cout)*size*size/S**2 ，计算出来的量约为118M

#### 需要注意的是：
**表中的计算量指的是卷积层的乘累加运算（MAC）次数，不包含pooling层的计算量以及relu的计算量等；**

**表中的参数量指的是卷积层weight参数量，不包含bias参数量；**

该网络的设计考虑到了计算效率和实用性，因此可以在单个设备上运行，甚至包括那些计算资源有限的设备，尤其是内存占用率较低的设备

(The network was designed with computational efficiency and practicality in mind, so that inference
can be run on individual devices including even those with limited computational resources, especially with low-memory footprint)

### 神经网络结构如下

![4](https://user-images.githubusercontent.com/28073374/134776153-2a3d7802-7b2e-4040-b62d-4e83609bf282.png)

## 6 Training Methodology
### 关于训练方式
**使用SGD，0.9momentum 每8轮降低百分4的学习率**

（Our training used asynchronous stochastic gradient descent with 0.9 momentum [17], fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs)

此外，我们发现，安德鲁·霍华德（Andrew Howard）[8]的光度畸变在一定程度上有助于防止过度拟合。在里面 此外，我们开始使用随机插值方法（双线性、面积、最近邻和立方， 以相同的概率）相对较晚地调整大小，并与其他超参数改变，因此我们无法确定最终结果是否受到他们的积极影响

（Also, we found that the
photometric distortions by Andrew Howard [8] were useful to combat overfitting to some extent. In
addition, we started to use random interpolation methods (bilinear, area, nearest neighbor and cubic,
with equal probability) for resizing relatively late and in conjunction with other hyperparameter
changes, so we could not tell definitely whether the final results were affected positively by their
use.）

### 7 ILSVRC 2014 Classification Challenge Setup and Results/ 8 ILSVRC 2014 Detection Challenge Setup and Results
**略**

### 9 Conclusions

改paper提供了一种似于预期的最优稀疏结构 ，通过现成的密集构建块是一种可行的方法，以此来改进神经网络的性能。与较浅和较宽的网络相比，该方法的主要优点是在计算需求适度增加的情况下，显著提高了训练质量。
