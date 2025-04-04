---
title: 知识蒸馏简述（Knowledge Distillation）
categories:
  - 算法总结
  
tags:
  - 蒸馏模型
  
mathjax: true
copyright: true
abbrlink: distillationmodel
date: 2021-09-16

---

## 1 背景
目前有很多复杂的模型可以来完成不同的任务，但是部署重量级模型的集成在许多情况下并不总是可行的。有时，你的单个模型可能太大，以至于通常不可能将其部署到资源受限的环境中。这就是为什么我们一直在研究一些模型优化方法 ——量化和剪枝

## 2 Softmax的故事
当处理一个分类问题时，使用 softmax 作为神经网络的最后一个激活单元是非常典型的用法。这是为什么呢？**因为softmax函数接受一组 logit 为输入并输出离散类别上的概率分布**。比如，手写数字识别中，神经网络可能有较高的置信度认为图像为1。不过，也有轻微的可能性认为图像为7。如果我们只处理像[1,0]这样的独热编码标签(其中1和0分别是图像为1和7的概率)，那么这些信息就无法获得。

<!--more-->

人类已经很好地利用了这种相对关系。更多的例子包括，长得很像猫的狗，棕红色的，猫一样的老虎等等。正如 Hinton 等人所认为的
>一辆宝马被误认为是一辆垃圾车的可能性很小，但比被误认为是一个胡萝卜的可能性仍然要高很多倍。

这些知识可以帮助我们在各种情况下进行极好的概括。这个思考过程帮助我们更深入地了解我们的模型对输入数据的想法。它应该与我们考虑输入数据的方式一致。

**而模型的 softmax 信息比独热编码标签更有用，因为本身的结果就是一种分布，人类认识世界又何尝不是如此**。

## 3 模型蒸馏的流程

* 在原始数据集上训练一个复杂但效果好的大模型，将此作为 `teacher model`；
* 基于教师模型在数据集上的预估结果 `soft label`，重新在数据集上训练一个轻量的模型，并尽量保留效果，此便是`student model`。

>最终目的是得到一个小而美的模型以便于在生产中进行部署使用。

本文以一个**图像分类**的例子，我们可以扩展前面的思想：

* 训练一个在图像数据集上表现良好的教师模型。
* 在这里，交叉熵损失将根据数据集中的真实标签计算。
* 在相同的数据集上训练一个较小的学生模型，但是使用来自教师模型(softmax输出)的预测作为 ground-truth 标签。

这些 softmax 输出称为软标签 `soft label`，原始的标签即为 `hard label`。

## 4 使用soft label的原理
在容量方面，我们的学生模型比教师模型要小。

因此，如果你的数据集足够复杂，那么较小的student模型可能不太适合捕捉训练目标所需的隐藏表示。我们在软标签上训练学生模型来弥补这一点，它提供了比独热编码标签更有意义的信息。在某种意义上，我们通过暴露一些训练数据集来训练学生模型来模仿教师模型的输出。

## 5 损失函数的构建
>实际中存在`弱概率`的问题是：它们没有捕捉到学生模型有效学习所需的信息。

例如，如果概率分布像[0.99, 0.01]，几乎不可能传递图像具有数字7的特征的知识。

Hinton 等人解决这个问题的方法是：**在将原始 logits 传递给 softmax 之前，将教师模型的原始 logits 按一定的温度进行缩放**。

这样，就会在可用的类标签中得到更广泛的分布。然后用同样的温度用于训练学生模型。

我们可以把学生模型的修正损失函数写成这个方程的形式：
$$L_{KDCE}=-\sum_i p_i \log s_i$$

其中，$p_i$是教师模型得到软概率分布，si的表达式为：
$$s_i=-\frac{exp(z_i/T)}{\sum_j exp(z_j/T)}$$

具体到代码的实现如下所示：
```python
def get_kd_loss(student_logits, teacher_logits, 
                true_labels, temperature,
                alpha, beta):
    
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kd_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature, 
        from_logits=True)
    
    return kd_loss
```

## 6 合并使用软硬标签
Hinton 等人还探索了在真实标签(通常是 one-hot 编码)和学生模型的预测之间使用传统交叉熵损失的想法。当训练数据集很小，并且软标签没有足够的信号供学生模型采集时，这一点尤其有用。

当它与扩展的 softmax 相结合时，这种方法的工作效果明显更好，而整体损失函数成为两者之间的加权平均。

$$L=\frac{\alpha * L_{KDCE}+\beta * L_{CE}}{(\alpha + \beta)}$$

其中
$$L_{CE}=-\sum_i y_i \log z_i$$

而$y_i$和$z_i$分别就是原始的标签即`hard label`和学生模型的原始预测结果。

具体代码实现可以如下所示：
```python
def get_kd_loss(student_logits, teacher_logits, 
                true_labels, temperature,
                alpha, beta):
    teacher_probs = tf.nn.softmax(teacher_logits / temperature)
    kd_loss = tf.keras.losses.categorical_crossentropy(
        teacher_probs, student_logits / temperature, 
        from_logits=True)
    
    ce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        true_labels, student_logits, from_logits=True)
    
    total_loss = (alpha * kd_loss) + (beta * ce_loss)
    return total_loss / (alpha + beta)
```

结合起来看便可以知道一个是基于软标签训练的，而另一个就是基于原始硬标签训练的。并且在实际使用中，**一般的$\alpha$取值要大于$\beta$比较好**，以便更多的吸收教师模型的信息。

## 7 直接拟合软标签
既然我们想学习教师模型的信息，最粗暴的做法便是以教师模型的结果`soft label`作为目标，直接进行回归。Caruana 等人便是如此，操作原始 logits，而不是 softmax 值。这个工作流程如下：

* 这部分保持相同:训练一个教师模型。这里交叉熵损失将根据数据集中的真实标签计算。
* 现在，为了训练学生模型，训练目标变成分别最小化来自教师和学生模型的原始对数之间的平均平方误差。

$$L_{KDMSE}=\sum_i||z_i^{\theta_student} - z_{i(teacher)}^{fixed}||^2$$

具体代码实现可如下所示：
```python
mse = tf.keras.losses.MeanSquaredError()

def mse_kd_loss(teacher_logits, student_logits):
    return mse(teacher_logits, student_logits)
```

使用这个损失函数的一个**潜在缺点是它是无界的**。原始 logits 可以捕获噪声，而一个小模型可能无法很好的拟合。这就是为什么为了使这个损失函数很好地适合蒸馏状态，学生模型需要更大一点。

Tang 等人探索了在两个损失之间插值的想法：**扩展 softmax 和 MSE 损失**。数学上，它看起来是这样的：
$$L=(1-\alpha) \cdot L_{KDMSE} + \alpha \cdot L_{KDCE}$$

根据经验，他们发现当 $\alpha = 0$ 时，(在NLP任务上)可以获得最佳的性能。

## 8 实践中的一些经验

### 8.1 使用数据增强
他们在NLP数据集上展示了这个想法，但这也适用于其他领域。为了更好地指导学生模型训练，使用数据增强会有帮助，特别是当你处理的数据较少的时候。因为我们通常保持学生模型比教师模型小得多，所以我们希望学生模型能够获得更多不同的数据，从而更好地捕捉领域知识。

### 8.2 使用未标记的数据
>在像 Noisy Student Training 和 SimCLRV2 这样的文章中，作者在训练学生模型时使用了额外的未标记数据。

因此，你将使用你的 teacher 模型来生成未标记数据集上的 ground-truth 分布。这在很大程度上有助于提高模型的可泛化性。**这种方法只有在你所处理的数据集中有未标记数据可用时才可行**。有时，情况可能并非如此(例如，医疗保健)。也有研究数据平衡和数据过滤等技术，以缓解在训练学生模型时合并未标记数据可能出现的问题。

### 8.3 在训练教师模型时不要使用标签平滑

`标签平滑`是一种技术，**用来放松由模型产生的高可信度预测**。它有助于减少过拟合，但不建议在训练教师模型时使用标签平滑，因为无论如何，它的 logits 是按一定的温度缩放的。因此，一般不推荐在知识蒸馏的情况下使用标签平滑。

### 8.4 使用更高的温度值
Hinton 等人建议使用更高的温度值来 soften 教师模型预测的分布，这样软标签可以为学生模型提供更多的信息。这在处理小型数据集时特别有用。对于更大的数据集，信息可以通过训练样本的数量来获得。

## 9 代码
具体的实现代码，可以参考[DushyantaDhyani](https://github.com/DushyantaDhyani/kdtf)代码，是比较简洁易懂的。
值得注意的是：
1. 其在训练教师模型的时候使用的是
```python
logits = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out']) / self.softmax_temperature
```

2. 在训练学生模型的时候，使用了
```python
self.total_loss += tf.square(self.softmax_temperature) * self.loss_op_soft
```
并不是单独定义$\alpha$和$\beta$的。

**参考文章**
[神经网络中的蒸馏技术，从Softmax开始说起](https://mp.weixin.qq.com/s/IAk61KBKgOBsx9X2zINlfg)
[Distilling Knowledge in Neural Networks](https://wandb.ai/authors/knowledge-distillation/reports/Distilling-Knowledge-in-Neural-Networks--VmlldzoyMjkxODk)

---