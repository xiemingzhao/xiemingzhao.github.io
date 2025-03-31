---
title: XGBoost原理细节详解
date: 2019-07-01
abbrlink: XGBoostDetailAnalysis
categories:
  - 学习笔记
tags:
  - XGBoost
  - 机器学习
  - 算法
copyright: true
mathjax: true
---

原文来自大神级的论文[XGBoost: A Scalable Tree Boosting System](https://netman.aiops.org/~peidan/ANM2018/3.MachineLearningBasics/LectureCoverage/18.xgboost.pdf)，论文很全面，框架介绍很完整，但是在某些tricks上面并没有对细节做详细解说，而需要读者亲自去进行一定的推导，这使得阅读起来稍显吃力，当然基础很雄厚的大牛级别的应该不以为然，但我相信还有很多与我一样入行不久的，那么这篇博客就是你的所需。

**这里特别感谢作者`meihao5`的博文，其分享的内容就是我一直想要整理但迟迟未进行的，它的原文可见最后面的参考文章链接里。**

## 1 基础知识
`XGBoost`的成功可以总结为**回归（树回归+线性回归）+提升（boosting）+优化（5个方面）牛顿法、预排序、加权分位数、稀疏矩阵识别以及缓存识别**等技术来大大提高了算法的性能。下面开始介绍一些入门必须的基础知识：

<!--more-->

### 1.2 低维到高维的转变
#### 梯度和Hessian矩阵

- 一阶导数和梯度(gradient vector)

$$f'(x); g(x) = \nabla f(x) = \frac{\partial f(x)}{\partial x} = \left[\begin{array} {c} \frac{\partial f(x)}{\partial x_1}\\ \vdots \\ \frac{\partial f(x)}{\partial x_n} \end{array} \right]$$

- 二阶导数和`Hessian`矩阵

$$f''(x); H(x) = \nabla f(x) = \left[\begin{array} {c c c c} \frac{\partial^2 f(x)}{\partial x_1^2} \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} \cdots \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f(x)}{\partial x_2 \partial x_1} \frac{\partial^2 f(x)}{\partial x_2^2} \cdots \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\ \vdots \\ \frac{\partial^2 f(x)}{\partial x_n \partial x_1} \frac{\partial^2 f(x)}{\partial x_n \partial x_2} \cdots \frac{\partial^2 f(x)}{\partial x_n^2} \end{array} \right]$$

### 1.3 泰勒级数和极值

**泰勒级数展开（标量和向量）**

- 输入为标量的泰勒级数展开

$$f(x_k + \delta) \approx f(x_k) + f'(x_k) \delta + \frac{1}{2} f''(x_k) \delta^2 + \cdots + \frac{1}{k!}f^k (x_k) \delta^k + \cdots$$

- 输入为向量的泰勒级数展开

$$f(x_k + \delta) \approx = f(x_k) + g^T (x_k) \delta + \frac{1}{2} \delta^T H(x_k) \delta$$

### 1.4 极值点

**标量情况**

- 输入为标量的泰勒展开

$$f(x_k + \delta) \approx f(x_k) + f'(x_k) \delta + \frac{1}{2} f''(x_k) \delta^2 $$

- 严格局部极小点指：$f(x_k + \delta) > f(x_k)$

- 称满足$f'(x_k) = 0$的点为平稳点（候选点）。
- 函数在$x_k$有严格局部极小值条件为$f'(x_k) = 0$且$f''(x_k) > 0$。

**向量情况**

- 输入为向量的泰勒级数展开

$$f(x_k + \delta) \approx f(x_k) +  g^T (x_k) \delta + \frac{1}{2} \delta^T H(x_k) \delta$$

- 称满足$g(x_k) = 0$的点为`平稳点`（候选点），此时如果有
>$H(x_k) \succ 0$， $x_k$为一个严格局部极小点（反之，局部严格最大点）

如果$H(x)$不定矩阵，是一个`鞍点`(saddle point)。（如下图所示）

![XGBoostDetailAnalysis1](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis1.png)

### 1.5 怎么求一个函数的极值

答案自然是`迭代法`。迭代法的基本结构可以表示成如下所示（最小化$f(x)$）：
1. 选择一个初始点，设置一个 convergence tolerance $\epsilon$，技术k=0
2. 决定搜索方向$d_k$， 使得函数下降（核心）
3. 决定步长$\alpha_k$是的$f(x_k + \alpha_k d_k)$对于$\alpha_k \geq 0$最小化，构建$x_{k+1} = x_k + \alpha_k d_k$
4. 如果$||d_k|| < \epsilon$，则停止输出解$x_{k+1}$，否则继续迭代。（如下图所示）

![XGBoostDetailAnalysis2](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis2.png)

**各种各样的优化算法不同点在于：选取的步长不一样，选取的方向不一样。**

Xgboost 也是 GBDT 的一种，只不过进行了大量的优化！其中一点就是优化方法选取了`牛顿法`（选取的方向不一样，一个梯度的方向，一个二阶导数的方向）

## 2 GBDT(梯度提升树）
如何构建得当的`回归提升树`（CRAT树），简单来说就是重复构建很多树，每一棵树都是基于前面的一棵树，使得当前这棵树拟合样本数据平方损失最小。

>当损失函数是平方损失函数或者指数函数时，每一步优化很简单，但是对一般损失函数，优化就不算那么容易了。于是，就有了梯度提升树算法。

**梯度提升算法的本质：拟合一个回归树是的损失函数最小**。
这个思想在优化算法经常用，但是没有解析解，一般就是拟合一个近似值（例如注明的著名的拟牛顿法）。

### 2.1 参数空间与函数空间

因为梯度提升树就是在函数空间做优化，如下图所示：

![XGBoostDetailAnalysis3](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis3.png)

### 2.2 Boosting思想

提升树使用了`Boosting`思想，即：
>先从初始训练集中训练出一个基学习器，再根据学习器的表现对训练样本分布进行调整，使得先前基学习器做错的样本在后续受到更多的关注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直到基学习器数目达到事先指定的T，最终将T个基学习器进行加权结合。

**Gradient Boosting Tree 算法原理**

- Friedman在论文[greedy function approximation: a gradient boosting machine](https://www.jstor.org/stable/2699986)中提出`GBDT`。

- 其模型F定义为`加法模型`：
  $$F(x;w) = \sum_{t=0}^T \alpha_t h_t(x;w_t) = \sum_{t=0}^T f_t (x;w_t)$$
  其中，x 为输入样本， h 为分类回归树，w 是分类回归树的参数，$\alpha$ 是每个树的权重。

- 通过最小化损失函数求解最优模型：
  $$F^* = \mathop{\arg\min}\limits_{F} \sum_{i=0}^N L(y_i, F(x_i; w))$$
  NP难问题 -> 通过贪心算法，迭代求局部最优解。

**计算流程表示如下：**
>输入：$(x_i, y_i), T, L$
1. 初始化$f_0$
2. for t=1 to T do
   2.1 计算响应：
   $\tilde y_i = -[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}]_{F(x) = F_{t-1}(x)}, i = 1,2,\cdots,N$
   2.2 学习第t棵树：
   $w^* = \mathop{\arg\min}\limits_{w} \sum_{i=1}^N(\tilde y_i - h_t(x_i;w))^2$
   2.3 line search 找步长（前向分步算法）：
   $\rho^* = \mathop{\arg\min}\limits_{\rho} \sum_{i=1}^N L(y_i,F_{t-1}(x_i) + \rho h_t(x_i;w^*))$
   2.4  令$f_t = \rho^* h_t(x;w^*)$，更新模型：
   $F_t = F_{t-1} + f_t$
3. 输出$F_T$

根据上述流程，类比梯度下降，自然有一些梯度提升的感觉，一个是优化参数空间，一个是优化函数空间。
1. 计算残差（计算值域真实值之间的误差）
2. 拟合是的残差最小（当前学习的这棵树）
3. $\rho$步长：基于学习器的权重， H树：表示方向
4. 得到当前这一步的树

**一句话总结：新树模型的引入是为了减少上个树的残差，即前面模型未能拟合的剩余信息。我们可以在残差减少的梯度方向上建立这么一个新模型。对比提升树来说，提升树没有基学习器参数权重$\rho$。**

以前面的均方损失为例，也是可以用这个方法来解释的。为了求导方便，我们在均方损失函数前乘上一个1/2：
$$L(y_i, F(x_i)) = \frac{1}{2} (y_i - F(x_i))^2$$

注意到$F(x_i)$其实只是一些数字而已，我们可以将其像变量一样进行求导：
$$\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)} = F(x_i) - y_i$$

而前面所说的残差就是上式相反数，即**负梯度**：
$$r_{ti} = y_i - F_{t-1}(x) = -[\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}]_{F(x) = F_{t-1}(x)}$$

随着T的增大我们的模型的训练误差会越来越小，如果无限迭代下去，理想情况下训练误差就会收敛到一个极小值，相应的会收敛到一个极小值点 P 。这是不是有种似曾相似的感觉，想一想在凸优化里面梯度下降法（参数空间的优化），是不是很像？我们就把F(x)看成是在 N 维空间中的一个一个的点，而损失函数就是这个N 维空间中的一个函数（函数空间的优化），我们要用某种逐步逼近的算法来求解损失函数的极小值（最小值）。


### 2.3 如果要将GBDT用于分类问题，怎么做呢？

首先要明确的是，**GBDT 用于回归使用的仍然是 CART 回归树**。

回想我们做回归问题的时候，每次对残差（负梯度）进行拟合。而分类问题要怎么每次对残差拟合？要知道类别相减是没有意义的。因此，可以用Softmax进行概率的映射，然后拟合概率的残差！

具体的做法如下：
1. 针对每个类别都先训练一个回归树，如三个类别，训练三棵树。就是比如对于样本$x_i$为第二类，则输入三棵树分别为：$(x_i,0),(x_i,1);(x_i,0)$这其实是典型的OneVsRest的多分类训练方式。
2. 而每棵树的训练过程就是CART的训练过程。这样，对于样本$x_i$就得出了三棵树的预测值$F1(x_i),F2(x_i),F3(x_i)$，模仿多分类的逻辑回归，用Softmax来产生概率，以类别1为例：
   $p1(x_i)=\frac{exp(F1(x_i))}{\sum_{i=1}^3 (F1(xi))}$

对每个类别分别计算残差，如
类别1：y~i1=0–p1(xi),
类别2：y~i2=1–p2(xi),
类别3：y~i3=0–p3(xi)

3. 开始第二轮的训练，针对第一类 输入为(xi,y~i1), 针对第二类输入为(xi,y~i2)针对第三类输入为(xi,y~i3)，继续训练出三颗树。

重复3直到迭代M轮，就得到了最后的模型。预测的时候只要找出概率最高的即为对应的类别。和上面的回归问题是大同小异的。

## 3 XGBoost
所有的机器学习的过程都是一个搜索假设空间的过程，我们的模型就是在空间中搜索一组参数（这组参数组成一个模型），使得和目标最接近（损失函数或目标函数最小），通过不断迭代的方式，不断的接近学习到真实的空间分布。

得到这样一个分布或者映射关系后，对空间里的未知样本或者新样本就可以做出预测/推理。这也解释了为什么一般样本越多模型效果越好，（大数定律）

**有多少人工就有多少智能！**

真实的样本空间是有噪声的，所以学习准确率不可能百分之百。（贝叶斯上限）

### 3.1 模型函数形式
给定数据集$\mathcal D = { (x_i, y_i) }$，XGBoost进行 additive training，学习 K 颗树，采用以下函数对样本进行预测：

$$\hat y_i = \phi (x_i) = \sum_{k=1}^K f_k (x_i), f_k \in \mathcal F$$

这里 $\mathcal F$ 是假设空间， $f(x)$ 是回归树（CART）：
$$\mathcal F = \{ f(x) = w_{q(x)} \} (q:\mathbb R^m \rightarrow T, w \rightarrow \mathbb R^T)$$

$q(x)$ 表示将样本 x 分到了某个叶子结点上， w 是叶子结点的分数(leaf score)， 所以 $w_{q(x)}$ 表示回归树对样本的预测值。

### 3.2 目标函数
参数空间中的`目标函数`：

$$Obj(\Theta) = L(\Theta) + \Omega (\Theta)$$

- $L(\Theta)$是误差函数，衡量模型拟合数据的程度；
- $\Omega (\Theta)$ 是正则化项，用来惩罚复杂模型的。

误差函数可以是 `square loss`， `log loss` 等，正则项可以是 L1 正则项，L2 正则等。

- Ridge Regression （岭回归）： $\sum_{i=1}^n (y_i - \theta^T x_i)^2 + \lambda ||\theta||^2$
- LASSO：$\sum_{i=1}^n (y_i - \theta^T x_i)^2 + \lambda ||\theta||_1$

### 3.3 正则项
正则项的作用，可以从几个角度去解释：

- 通过偏差方差分解去解释
- PAC-learning 泛化界解释
- Bayes 先验解释，把正则当成先验

从 Bayes 角度来看，正则相当于对模型参数引入先验分布：

![XGBoostDetailAnalysis4](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis4.PNG)

- L2 正则中，模型参数服从搞死分布 $\theta ~ N(0,\sigma^2)$， 对参数加了分布约束，大部分绝对值很小；
- L1 正则中， 模型参数服从拉普拉斯分布， 对参数加了分布约束，大部分取值为0。

### 3.4 XGBoost 的目标函数

#### 正则项
XGBoost 的目标函数（函数空间）：
$$\mathcal L(\phi) = \sum_i l(\hat y_i, y_i) + \sum_k \Omega (f_k)$$

其中正则项对每棵树的复杂度进行了惩罚。

相比原始的 GBDT， XGBoost 的目标函数多了正则项， 是的学习出来的模型更加不容易过拟合。

有哪些指标可以衡量树的复杂度？
**树的深度，内部节点个数，叶子节点个数（T）， 叶子节点分数（w）...**

XGBoost 采用的是：
$$\Omega (f) = \gamma T + \frac{1}{2} \lambda ||w||^2$$
对叶子节点个数进行了惩罚， 相当于在训练过程中做了剪枝。

**怎么求最小目标函数？**
GBDT 是通过求一阶导数，迭代法的方式在函数空间拟合一个最小值。 XGBoost 通过泰勒展开实现了更精确的拟合。

### 3.5 误差函数的二阶泰勒展开
- 第 t 次迭代后， 模型的预测等于前 t-1 次的模型预测加上第 t 颗树的预测：
  $$\hat y_i^{(t)} = \hat y_i^{(t-1)} + f_t (x_i)$$

- 此时目标函数可写作：
  $$\mathcal L ^{(t)} = \sum_{i=1}^n l(y_i, \hat y_i^{(t-1)} + f_t(x_i)) + \Omega (f_t)$$
  公式中 $y_i, \tilde y_i^{(t-1)}$都已知， 模型要学习的只有第 t 颗树$f_t$

- 将误差函数在 $\tilde y_i^{(t-1)}$ 处二阶泰勒展开：
  $$\mathcal L^{(t)} \simeq \sum_{i=1}^n [l(y_i, \hat y^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2 (x_i)] + \Omega (f_t)$$

公式中，$g_i = \partial_{\hat y^{(t-1)} } l(y_i, \hat y^{(t-1)}), h_i = \partial_{\hat y^{(t-1)} }^2 l(y_i, \hat y^{(t-1)})$

- 将公式中的常数项去掉，得到：
  $$\mathcal{\tilde L^{(t)} } = \sum_{i=1}^n [ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2 (x_i)] + \Omega (f_t)$$

- 把 $f_t, \Omega(f_t)$ 写成树结构的形式， 即把下式带入目标函数中：
  $$f(x) = w_{q(x)}, \Omega (f) = \gamma T + \frac{1}{2} \lambda ||w||^2$$

得到：
$$
\begin{array}{l}
\mathcal{\tilde L^{(t)} } = \sum_{i=1}^n [ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2 (x_i)] + \Omega (f_t) \\
= \sum_{i=1}^n [g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2] + \gamma T + \lambda \frac{1}{2} \sum_{j=1}^T w_j^2
\end{array}
$$

- $\sum_{i=1}^n [g_i w_{q(x_i)} + \frac{1}{2} h_i w_{q(x_i)}^2]%=$ 是对样本的累加， $\frac{1}{2} \sum_{j=1}^T w_j^2$ 是对叶节点的累加。

- 如何统一呢？定义每个叶节点 j 上的样本集合为 $I_j = \{ i | q(x_i) = j \}$，则目标函数可以写成按叶节点累加的形式：

$$\begin{array}{c}
\mathcal{\tilde L^{(t)}} = \sum_{j=1}^T[(\sum_{i \in I_j} g_i) w_j + \frac{1}{2} (\sum_{i \in I_j} h_i + \lambda) w_j^2] + \gamma T \\
= \sum_{j=1}^T[G_j w_j + \frac{1}{2}(H_j + \lambda) w_j^2] + \gamma T
\end{array}$$

- 如果确定了树的结构（即 $q(x)$ 确定了）， 为了使目标函数最小，可以令其导数为0， 解得每个叶节点的最优预测分数为：
  $$w_j^* = - \frac{G_j}{H_j + \lambda}$$

带入目标函数，得到最小损失为：
$$\mathcal{\tilde L^*} = -\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$

### 3.6 回归树的学习策略
>当回归树的结构确定时，我们前面已经退到出其最优的叶节点分数以及对应的最小损失值，问题是怎么确定树的结构？

- 暴力枚举所有可能的树结构，选择损失值最小的 - NP 难问题
- 贪心法， 每次尝试分裂一个叶节点，计算分裂前后的增益，选择增益最大的。

**分裂前后的增益怎么计算呢？**

- ID3 算法采用信息增益
- C4.5 算法采用信息增益比
- CART 采用 Gini 系数
- XGB 不一致

### 3.7 XGBoost 的打分函数

$$\mathcal{\tilde L^*} = - \frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j + \lambda} + \gamma T$$

部分衡量了每个叶子节点对总体损失的贡献， 我们希望损失越小越好， 则前半部分的值越大越好。

因此， 对一个叶子结点进行分裂，分裂前后的`增益`定义为：

$$Gain = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R+ \lambda} - \gamma$$

`Gain` 值越大，分裂后 L 减小越多。所以当对一个叶节点分割时，计算所有候选（feature, value）对应的 gain， 选取 gain 最大的进行分割。

这个公式跟我们之前遇到的信息增益或基尼值增量的公式是一个道理。XGBoost 就是利用这个公式计算出的值作为分裂条件。

**分裂后左边增益+右边增益-分类前增益**
也就是`最大损失减小值`的原则来选择。

### 3.8 树节点分裂算法

- 近似算法距离：三分位数

![XGBoostDetailAnalysis5](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis5.PNG)

如上图所示，

$$
\begin{array}{l}
Gain = max\{ Gain, \frac{G_1^2}{H_1 + \lambda} + \frac{G_{23}^2}{H_{23} + \lambda} - \frac{G_{123}^2}{H_{123} + \lambda} - \gamma, \\
\frac{G_{12}^2}{H_{12} + \lambda} + \frac{G_3^2}{H_3 + \lambda} - \frac{G_{123}^2}{H_{123} + \lambda} - \gamma\}
\end{array}$$

- 实际上 XGBoost 不是简单按照样本个数进行分位， 而是以二阶导数值作为权重（Weighted Quantile Sketch）， 比如：

![XGBoost-Detail-Analysis6.png](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis6.PNG)

- 为什么用 $h_i$ 加权，就是把目标函数整理成以下形式，可以看出 $h_i$ 有对 loss 加权的作用。

$$\sum_{i=1}^n \frac{1}{2} h_i (f_t(x_i)) - g_i/h_i)^2 + \Omega (f_t) + constant$$

### 3.9 稀疏值处理

- 稀疏值：缺失导致，诸如类别类 one-hot 编码会导致大量 0 值出现。
- 当特征出现缺失值的时候 XGBoost 可以学习出默认的节点分裂方向，如下图算法所示：

![XGBoostDetailAnalysis7](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis7.png)

**不会对该特征为missing的样本进行遍历统计，只对该列特征值为 non-missing 的样本上对应的特征值进行遍历**


#### 最后一步：
通过上述算法经过T此迭代我们得到T+1个弱学习器，$\{ F(x)_0,F(x)_1,F(x)_2, \cdots \}$

那么通什么样的形式将他们迭代起来呢？答案是直接将 T+1个 模型相加，只不过为了防止过拟合，XGBoost 也采用了 shrinkage 方法来降低过拟合的风险，其模型集成形式如下：

$$F_m(X) = F_{m-1}(X) + \eta f_m(X), 0< \eta \leq 1$$

Shrinkage 论文提到：关于 n 和迭代次数 T 的取值，可以通过交叉验证得到合适的值，通常针对不同问题，其具体值是不同的。一般来说，当条件允许时（如对模型训练时间没有要求等）可以设置一个较大的迭代次数 T ，然后针对该 T 值利用交叉验证来确定一个合适的 n 值。但 n 的取值也不能太小，否则模型达不到较好的效果.


## 4 更多特性

### 4.1 XGBoost 的其他特性

- 行抽样（row sample）
- 列抽样（column sample），借鉴随机森林
- Shrinkage（缩减）， 即学习速率
  将学习速率调小，迭代次数增多，有正则化作用
- 支持自定义损失函数（需二阶可导）

![XGBoostDetailAnalysis8](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis8.png)

### 4.2 XGBoost 的系统设计

- Column Block
1. 特征预排序，以 column block 的结构存于内存中
2. 存储样本索引（instance indices）
3. block 中的数据以稀疏格式（CSC）存储

这个结构加速了 `split finding` 的过程， 只需要在建树前排序一次，后面节点分裂时直接根据索引得到梯度信息

- Cache Aware Access
1. column block 按特征大小顺序存储， 相应的样本的梯度信息是分散的，造成内存的不连续访问，降低 CPU cache 命中率
2. 缓存优化方法
- 预取数据到buffer 中（非连续->连续）， 在统计梯度信息
- 调节块的大小

![XGBoostDetailAnalysis9](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis9.png)

### 4.3 更高效的工具包 LightGBM

- 速度更快
  ![XGBoostDetailAnalysis10](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis10.png)

- 内存占用更低
  ![XGBoostDetailAnalysis11](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/XGBoostDetailAnalysis11.png)

- 准确率更高（优势不明显， 与 XGBoost 相当）
  *在微软的论文中说是改动很大，实际应用中没有那么明显，可能与数据集有关系*

**主要改进：直方图优化，进一步并行优化。**


### 4.4 XGBoost 的参数意义与调优：

1）Booster: 分类器类型
2）lambda: 正则化
3）min_child_weight:子节点权重
4）树的深度
4）学习率n
......

## XGBoost总结：

1. 损失函数是用泰勒展式二项逼近，而不是像GBDT里的就是一阶导数；
2. 对树的结构进行了正则化约束，防止模型过度复杂，降低了过拟合的可能性；
3. 实现了并行化（树节点分裂的时候）
4. 开头提到的优化
5. 回归模型可选


**参考文章**
1. [xgboost原理详解-meihao5](https://blog.csdn.net/meihao5/article/details/83788525)
2. [XGBoost- A Scalable Tree Boosting System](http://delivery.acm.org/10.1145/2940000/2939785/p785-chen.pdf?ip=61.152.150.141&id=2939785&acc=CHORUS&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E6D218144511F3437&__acm__=1576504271_55fd2b06af4e72ca559df3a74156a91f)

---