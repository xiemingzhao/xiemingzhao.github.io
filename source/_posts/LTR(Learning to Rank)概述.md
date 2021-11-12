---
title: LTR(Learning to Rank)概述
categories:
- 学习笔记
- 算法总结
tags:
- 机器学习
- 排序
- LTR
mathjax: true
copyright: true
abbrlink: IntroductionofLTR
date: 2019-06-29
---

## 1 Learning to Rank 简介
Learning to Rank是采用机器学习算法，通过训练模型来解决排序问题，在Information Retrieval，Natural Language Processing，Data Mining等领域有着很多应用。

## 1.1 排序问题
如图 Fig.1 所示，在信息检索中，给定一个query，搜索引擎会召回一系列相关的Documents（通过term匹配，keyword匹配，或者semantic匹配的方法），然后便需要对这些召回的Documents进行排序，最后将Top N的Documents输出,一版可以认为是召回后的精排。而排序问题就是使用一个模型 f(q,d)来对该query下的documents进行排序，这个模型可以是人工设定一些参数的模型，也可以是用机器学习算法自动训练出来的模型。现在第二种方法越来越流行，尤其在Web Search领域，因为在Web Search 中，有很多信息可以用来确定query-doc pair的相关性，而另一方面，由于大量的搜索日志的存在，可以将用户的点击行为日志作为training data，使得通过机器学习自动得到排序模型成为可能。

**需要注意的是，排序问题最关注的是各个Documents之间的相对顺序关系，而不是各个Documents的预测分最准确。**

<!--more-->

Learning to Rank是监督学习方法，所以会分为training阶段和testing阶段，如图 Fig.2  所示。

<center class="half">
    <img src="http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-1.png" width="400"/><img src="http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-2.png" width="400"/>
</center>

### 1.1.1 Training data的生成
对于Learning to Rank，training data是必须的，而feature vector通常都是可以得到的，关键就在于label的获取，而这个label实际上反映了query-doc pair的真实相关程度。通常我们有两种方式可以进行label的获取：

- 第一种方式是**人工标注**，这种方法被各大搜索引擎公司广为应用。人工标注即对抽样出来作为training data的query-doc pair人为地进行相关程度的判断和标注。一般标注的相关程度分为5档：perfect，excellent，good，fair，bad。例如，query=“Microsoft”，这时候，Microsoft的官网是perfect；介绍Microsoft的wikipedia则是excellent；一篇将Microsoft作为其主要话题的网页则是good；一篇只是提到了Microsoft这个词的网页则是fair，而一篇跟Microsoft毫不相关的网页则是bad。人工标注的方法可以通过多人同时进行，最后以类似投票表决的方式决定一个query-doc pair的相关程度，这样可以相对减少各个人的观点不同带来的误差。

- 第二种方式是**通过搜索日志获取**。搜索日志记录了人们在实际生活中的搜索行为和相应的点击行为，点击行为实际上隐含了query-doc pair的相关性，所以可以被用来作为query-doc pair的相关程度的判断。一种最简单的方法就是利用同一个query下，不同doc的点击数的多少来作为它们相关程度的大小。实际中一般使用行为/曝光来制作target，但是最终还是要将连续型的值进行分桶，否则泛化能力不够。

不过需要注意的是，这里存在着一个很大的陷阱，就是用户的点击行为实际上是存在**“position bias”**的，即用户偏向于点击位置靠前的doc，即便这个doc并不相关或者相关性不高。有很多 tricky的和 general 的方法可以用来去除这个“position bias”，例如，

>1. 当位置靠后的doc的点击数都比位置靠前的doc的点击数要高了，那么靠后的doc的相关性肯定要比靠前的doc的相关性大。
2. Joachims等人则提出了一系列去除bias的方法，例如 Click > Skip Above, Last Click > Skip Above, Click > Earlier Click, Click > Skip Previous, Click > No Click Next等。
3. 有个很tricky但是效果很不错的方法，之前我们说一个doc的点击数比另一个doc的点击数多，并不一定说明前者比后者更相关。但如果两者的差距大到一定程度了，即使前者比后者位置靠前，但是两者的点击数相差5-10倍，这时候我们还是愿意相信前者更加相关。当然这个差距的大小需要根据每个场景具体的调整。
4. position bias 存在的原因是，永远无法保证在一次搜索行为中，用户能够看到所有的结果，往往只看到前几位的结果。这时候就到了 Click Model大显身手的时候了，一系列的 Click Model 根据用户的点击信息对用户真正看到的doc进行“筛选”，进而能更准确地看出用户到底看到了哪些doc，没有看到哪些doc，一旦这些信息知道了，那么我们就可以根据相对更准确的 点击数/展示数（即展现CTR）来确定各个doc的相关性大小。

上述讲到的两种label获取方法各有利弊。人工标注受限于标注的人的观点，不同的人有不同的看法，而且毕竟标注的人不是真实搜索该query的用户，无法得知其搜索时候的真实意图；另一方面人工标注的方法代价较高且非常耗时。而从搜索日志中获取的方法则受限于用户点击行为的噪声，这在长尾query中更是如此，且有用户点击的query毕竟只是总体query的一个子集，无法获取全部的query下doc的label。

### 1.1.2 Feature的生成
这里只是简单介绍下，后续博客会有更纤细的讲解。

一般Learning to Rank的模型的feature分为两大类：**relevance 和 importance（hotness）**，即query-doc pair 的相关性feature，和doc本身的热门程度的feature。两者中具有代表性的分别是 BM25 和 PageRank。

### 1.1.3 Evaluation
怎么判断一个排序模型的好坏呢？我们需要有验证的方法和指标。方法简单来说就是，比较模型的输出结果，和真实结果（ground truth）之间的差异大小。*用于Information Retrieval的排序衡量指标通常有：NDCG，MAP等。*

**NDCG（Normalized Discounted Cumulative Gain）：**
NDCG表示了从第1位doc到第k位doc的“归一化累积折扣信息增益值”。其基本思想是：
>1） 每条结果的相关性分等级来衡量
2） 考虑结果所在的位置，位置越靠前的则重要程度越高
3） 等级高（即好结果）的结果位置越靠前则值应该越高，否则给予惩罚

$$NDCG(k)=G_{max,i}^{-1}(k)\sum_{j:\pi_i(j) \leq k}G(j)D(\pi_i(j))$$

其中G表示了这个doc得信息增益大小，一般与该doc的相关程度正相关：

$$G(j)=2^{y_{i,j}}-1$$

D则表示了该doc所在排序位置的折扣大小，一般与位置负相关：

$$D(\pi_i(j))=\frac{1}{log_2(1+\pi_i(j))}$$

而$G_{max}$则表示了归一化系数，是最理想情况下排序的“累积折扣信息增益值”。
最后，将每个query下的NDCG值平均后，便可以得到排序模型的总体NDCG大小。

**MAP（Mean Average Precision）**：

其定义是求每个相关文档检索出后的准确率的平均值（即Average Precision）的算术平均值（Mean）。这里对准确率求了两次平均，因此称为Mean Average Precision。

在MAP中，对query-doc pair的相关性判断只有两档：1和0。
对于一个query，其AP值为：

$$AP=\frac{\sum_{j=1}^{n_i} P(j) \cdot y_{i,j} }{\sum_{j=1}^{n_i}y_{i,j} }$$

$y_{ij}$即每个doc的label（1和0），而每个query-doc pair的P值代表了到dij这个doc所在的位置为止的precision：

$$P(j)=\frac{\sum_{k:\pi_i(k)\leq \pi_i(j)}y_{i,k} }{\pi_i(j)}$$

其中，$\pi_i(j)$是$d_{ij}$在排序中的位置。

## 1.2 Formulation

用通用的公式来表示Learning to Rank算法，loss function为$L(F(x),y)$，从而risk function（loss function在X，Y联合分布下的期望值）为：

$$R(F)=\int_{ {\cal X} ,{\cal Y} }L(F(x),y) {\cal d}P(x,y)$$

有了training data后，进一步得到empirical risk function：

$$\hat R(F)=\frac{1}{m} \sum_{i=1}^m L'(F(x_i),y_i)$$

于是，学习问题变成了如何最小化这个empirical risk function。而这个优化问题很难解决，因为loss function不连续。于是可以使用一个方便求解的surrogate function来替换原始loss function，转而优化这个替换函数：

$$\hat {R'}(F)=\frac{1}{m} \sum_{i=1}^m L'(F(x_i),y_i)$$

替换函数的选择有很多种，根据Learning to Rank的类型不同而有不同的选择：

1）pointwise loss：例如squared loss等。

$$L'(F(x),y)=\sum_{i=1}^n (f(x_i),y_i)^2$$

2）pairwise loss：例如hinge loss，exponential loss，logistic loss等。

$$L'(F(x),y)=\sum_{i=1}^{n-1} \sum_{j=i+1}^n \phi(sign(y_i-y_j),f(x_i)-f(x_j))$$

3）listwise loss：

$$L'(F(x),y)=exp(-NDCG)$$

## 1.3 Learning to Rank Methods
Learning to Rank 方法可以分为三种类型：pointwise，pairwise，和listwise。

pointwise和pairwise方法将排序问题转化为classification，regression，ordinal classification等问题，优点是可以直接利用已有的classificatin和regression算法，缺点是group structure其实是被忽略的，即不会考虑每个query下所有doc之间的序关系。导致其学习目标和真实的衡量排序的目标并不一定是一致的（很多排序衡量指标，例如NDCG都是衡量每个query下的整体list的序关系的）。而listwise方法则将一个ranking list作为一个instance来进行训练，其实会考虑每个query下所有doc之间的序关系的。

这三种类型的Learning to Rank方法的具体算法一般有：

>1) Pointwise: Subset Ranking, McRank, Prank, OC SVM
2) Pairwise: Ranking SVM, RankBoost, RankNet, GBRank, IR SVM, Lambda Rank, LambdaMart
3) Listwise: ListNet, ListMLE, AdaRank, SVM MAP, Soft Rank

针对各个具体的算法介绍，后续的博客会进一步给出，这里就不再多加详述了。

---

## 2 RankNet，LambdaRank，LambdaMart简介

## 2.1 RankNet
RankNet是2005年微软提出的一种pairwise的Learning to Rank算法，它从概率的角度来解决排序问题。RankNet的核心是提出了一种概率损失函数来学习Ranking Function，并应用Ranking Function对文档进行排序。这里的Ranking Function可以是任意对参数可微的模型，也就是说，该概率损失函数并不依赖于特定的机器学习模型，在论文中，RankNet是基于神经网络实现的。除此之外，GDBT等模型也可以应用于该框架。

### 2.1.1 相关性概率
我们先定义两个概率：预测相关性概率、真实相关性概率。

**（1）预测相关性概率**
对于任意一个doc对$(U_i,U_j)$，模型输出的score分别为$s_i$和$s_j$，那么根据模型的预测，$U_i$比$U_j$与Query更相关的概率为：

$$P_{ij}=P(U_i>U_j)=\frac{1}{1+e^{-\sigma (s_i-s_j)}}$$

由于RankNet使用的模型一般为神经网络，根据经验sigmoid函数能提供一个比较好的概率评估。参数σ决定sigmoid函数的形状，对最终结果影响不大。

>RankNet证明了如果知道一个待排序文档的排列中相邻两个文档之间的排序概率，则通过推导可以算出每两个文档之间的排序概率。因此对于一个待排序文档序列，只需计算相邻文档之间的排序概率，不需要计算所有pair，减少计算量。

**（2）真实相关性概率**
对于训练数据中的$U_i$和$U_j$，它们都包含有一个与Query相关性的真实label，比如$U_i$与Query的相关性label为good，$U_j$与Query的相关性label为bad，那么显然$U_j$比$U_j$更相关。我们定义$U_j$比$U_j$更相关的真实概率为：

$$\bar P_{ij}=\frac{1}{2}(1+S_{ij})$$

如果$U_i$比$U_j$更相关，那么$S_{ij}$=1；如果$U_i$不如$U_j$相关，那么$S_{ij}$=−1；如果$U_i$、$U_j$与Query的相关程度相同，那么$S_{ij}$=0。通常，两个doc的relative relevance judgment可由人工标注或者从搜索日志中获取得到。

### 2.1.2 损失函数
对于一个排序，RankNet从各个doc的相对关系来评价排序结果的好坏，排序的效果越好，那么有错误相对关系的pair就越少。所谓错误的相对关系即如果根据模型输出$U_i$排在$U_j$前面，但真实label为$U_i$的相关性小于$U_j$，那么就记一个错误pair，RankNet本质上就是以错误的pair最少为优化目标。而在抽象成cost function时，**RankNet实际上是引入了概率的思想：不是直接判断$U_i$排在$U_j$前面，而是说$U_i$以一定的概率P排在$U_j$前面，即是以预测概率与真实概率的差距最小作为优化目标**。最后，RankNet使用Cross Entropy作为cost function，来衡量$P_{ij}$对$\bar P_{ij}$的拟合程度：

$$C=-\bar P_{ij}logP_{ij}-(1-\bar P_{ij})log(1-P_{ij})$$

带入相应等式整理得：
$$
\begin{align*}
C_{ij} &= -\frac{1}{2}(1+S_{ij})log \frac{1}{1+e^{-\sigma(s_i-s_j)}} -\frac{1}{2}(1-S_{ij}) log \frac{e^{-\sigma (s_i-s_j)}}{1+e^{-\sigma (s_i-s_j)}} \\
&=-\frac{1}{2}(1+S_{ij})log \frac{1}{1+e^{-\sigma(s_i-s_j)}}-\frac{1}{2}(1-S_{ij})[-\sigma(s_i-s_j)+log \frac{1}{1+e^{-\sigma(s_i-s_j)}}] \\
&=\frac{1}{2}(1-S_{ij})\sigma(s_i-s_j)+log(1+e^{-\sigma(s_i-s_j)})
\end{align*}
$$

其中：
$$
\begin{align*}
C=\left\{
\begin{array}{lr}
log(1+e^{-\sigma(s_i-s_j)}), & S_{ij}=1  \\
log(1+e^{-\sigma(s_j-s_i)}), & S_{ij}=-1 \\
\end{array}
\right.
\end{align*}
$$

下面展示了当$S_{ij}$分别取1，0，-1的时候cost function以$s_i-s_j$为变量的示意图：

![cost function.jpg](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-3.png)

可以看到当$S_{ij}$=1时，模型预测的$s_i$比$s_j$越大，其代价越小；$S_{ij}$=−1时，$s_i$比$s_j$j越小，代价越小；$S_{ij}$=0时，代价的最小值在$s_i$与$s_j$相等处取得。

该损失函数有以下几个特点：

>1) 当两个相关性不同的文档算出来的模型分数相同时，损失函数的值大于0，仍会对这对pair做惩罚，使他们的排序位置区分开。
2) 损失函数是一个类线性函数，可以有效减少异常样本数据对模型的影响，因此具有鲁棒性。

所以一个query的总代价为：

$$C=\sum_{(i,j)\in I}C_{ij}$$

其中，I表示所有在同一query下，且具有不同relevance judgment的doc pair，每个pair有且仅有一次。

### 2.1.3 合并概率
上述的模型$P_{ij}$需要保持一致性，即如果Ui的相关性高于$U_j$，$U_j$的相关性高于$U_k$，则Ui的相关性也一定要高于$U_k$。否则，如果不能保持一致性，那么上面的理论就不好使了。

我们使用$U_i$ vs $U_j$的真实概率 和 $U_j$ vs $U_k$ 的真实概率，计算$U_j$ vs $U_k$的真实概率：

$$\bar P_{ik}=\frac{\bar P_{ij}\bar P_{jk}}{1+2\bar P_{ij}\bar P_{jk}-\bar P_{ij}-\bar P_{jk}}$$

若$\bar P_{ij}=\bar P_{jk}=P$,则有如下图所示：

![$P_{ik}$变化图.jpg](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-4.png)

>1. $P=0$时，有$\bar P_{i,k}=P=0$表示：$D_i$排$D_j$后面,$D_j$排$D_k$的后面，则$D_i$一定排$D_k$的后面；
2. $0<P<0.5$时，\bar P_{i,k} < P$；
3. $P=0.5$时，有\bar P_{i,k} = P = 0.5$表示：$D_i$有一半概率排在$D_j$前面，$D_j$也有一半概率排在$D_k$的前面，则$D_i$同样也是一半的概率排在$D_k$的前面；
4. $0.5 < P <1$时，$\bar P_{i,k}>P$；
5. $P=1$时，有$\bar P_{i,k}=P=1$表示：$D_i$排在$D_j$前面，$D_j$排在$D_k$的前面，则$D_i$也一定排在$D_k$的前面；

### 2.1.4 Gradient Descent
我们获得了一个可微的代价函数，下面我们就可以用随机梯度下降法来迭代更新模型参数$w_k$了，即
$$w_k \rightarrow w_k - \eta \frac{\partial C}{\partial w_k}$$

$\eta$为步长，代价C沿负梯度方向变化。

$$\Delta =\sum_k \frac{\partial C}{\partial w_k} \delta w_k = \sum_k\frac{\partial C}{\partial w_k}(\eta \frac{\partial C}{\partial w_k})=-\eta \sum_k (\frac{\partial C}{\partial w_k})^2<0$$

这表明沿负梯度方向更新参数确实可以降低总代价。而使用了随机梯度下降法时，有：

$$
\begin{align*}
\frac{\partial C}{\partial w_k} &= \frac{\partial C}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C}{\partial s_j} \frac{\partial s_j}{\partial w_k} \\
&= \sigma (\frac{1}{2}(1-S_{ij})-\frac{1}{1 + e^{\sigma (s_i - s_j)}})(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}) \\
&=\lambda_{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}) \\
\end{align*}
$$

其中：

$$\lambda_{ij}=\frac{\partial C(s_i-s_j)}{\partial s_i} = \sigma (\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma (s_i-s_j)}})$$

### 2.1.5 加速RankNet训练过程
上面的是对于每一对pair都会进行一次权重的更新，其实是可以对**同一个query下的所有文档pair**全部带入神经网络进行前向预测，然后计算总差分并进行误差后向反馈，这样将大大减少误差反向传播的次数。

即，我们可以转而利用**批处理的梯度下降法**：

$$\frac{\partial C}{\partial w_k}=\sum_{(i ,j) \in I}(\frac{\partial C_{ij}}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C_{ij}}{\partial s_j} \frac{\partial s_j}{\partial w_k})$$

其中：
$$\frac{\partial C_{ij}}{\partial s_i}=\sigma (\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma (s_i-s_j)}})=-\frac{\partial C_{ij}}{\partial s_j}$$

令：
$$\lambda_{ij}=\frac{\partial C_{ij}}{\partial s_i} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma (s_i-s_j)}})$$

于是有：
$$
\begin{align*}
\frac{\partial C}{\partial w_k} &= \sum_{(i,j) \in I}\sigma (\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma (s_i-s_j)}})(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k}) \\
&=\sum_{(i,j) \in I} \lambda_{ij}(\frac{\partial s_i}{\partial w_k}-\frac{\partial s_j}{\partial w_k})\\
&=\sum_i \lambda_i \frac{\partial s_i}{\partial w_k} \\
\end{align*} \\
$$

下面我们来看看这个$\lambda_i$是什么。前面讲过集合I中只包含label不同的doc的集合，且每个pair仅包含一次，即$(U_i,U_j)$与$(U_j,U_i)$等价。为方便起见，我们假设I中只包含$(U_i,U_j)$)表示$U_i$相关性大于$U_j$的pair，即I中的pair均满足$S_{ij}=1$，那么

$$\lambda_i=\sum_{j:(i,j)\in I}\lambda_{ij}-\sum_{j:(j,i)\in I}\lambda_{ij}$$

这个写法是Burges的paper上的写法。下面我们用一个实际的例子来看：有三个doc，其真实相关性满足$U_1>U_2>U_3$，那么集合I中就包含{(1,2), (1,3), (2,3)}共三个pair，那么:
$$\frac{\partial C}{\partial w_k}=(\lambda_{12} \frac{\partial s_1}{\partial w_k}-\lambda_{12}\frac{\partial s_2}{\partial w_k})+(\lambda_{13}\frac{\partial s_1}{\partial w_k}-\lambda_{13}\frac{\partial s_3}{\partial w_k})+(\lambda_{23}\frac{\partial s_2}{\partial w_k}-\lambda_{23}\frac{\partial s_3}{\partial w_k})$$

显然$\lambda_1=\lambda_{12}+\lambda_{13},\lambda_2=\lambda_{23}-\lambda_{12},\lambda_3=-\lambda_{13}-\lambda_{23}$,因此$\lambda_i$其实可以写为：

$$\lambda_i=\sum_{j:(i,j)\in I}\lambda_{ij}-\sum_{k:(k,i)\in I}\lambda_{ki}$$

>**$\lambda_i$决定着第i个doc在迭代中的移动方向和幅度，真实的排在$U_i$前面的doc越少，排在$U_i$后面的doc越多，那么文档$U_i$向前移动的幅度就越大(实际$\lambda_i$负的越多越向前移动)。这表明每个f下次调序的方向和强度取决于同一Query下可以与其组成relative relevance judgment的“pair对”的其他不同label的文档。**
同时，这样的改造相当于是mini-batch learning。可以加速RankNet的学习过程。
原先使用神经网络模型，通过Stochastic gradient descent计算的时候，是对每一个pair对都会进行一次权重的更新。而通过因式分解重新改造后，现在的mini-batch learning的方式，是对同一个query下的所有doc进行一次权重的更新。时间消耗从O(n2)降到了O(n)。这对训练过程的影响是很大的，因为使用的是神经网络模型，每次权重的更新迭代都需要先进行前向预测，再进行误差的后向反馈。

## 2.2 Information Retrieval的评价指标
Information Retrieval的评价指标包括：MRR，MAP，ERR，NDCG等。NDCG和ERR指标的优势在于，它们对doc的相关性划分多个（>2）等级，而MRR和MAP只会对doc的相关性划分2个等级（相关和不相关）。并且，这些指标都包含了doc位置信息（给予靠前位置的doc以较高的权重），这很适合于web search。然而，这些指标的缺点是不平滑、不连续，无法求梯度，如果将这些指标直接作为模型评分的函数的话，是无法直接用梯度下降法进行求解的。

这里简单介绍下ERR（Expected Reciprocal Rank）。ERR是受到cascade model的启发，即一个用户从上到下依次浏览doc，直至他找到一个满意的结果，ERR可以定义为：

$$\sum_{r=1}^n \frac{1}{r}R_r \prod_{i=1}^{r-1}(1-R_i)$$

其中，$R_i$表示第i位的doc的相关性概率：

$$R_i=\frac{2^{l_i}-1}{2^{l_m}}$$

其中，$l_m$表示相关性评分最高的一档。

## 2.3 LambdaRank
上面我们介绍了以错误pair最少为优化目标的RankNet算法，然而许多时候仅以错误pair数来评价排序的好坏是不够的，像NDCG或者ERR等评价指标就只关注top k个结果的排序，当我们采用RankNet算法时，往往无法以这些指标为优化目标进行迭代，所以RankNet的优化目标和IR评价指标之间还是存在gap的。以下图为例：

![lambdarank.ipg](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-5.png)

如上图所示，每个线条表示文档，蓝色表示相关文档，灰色表示不相关文档，RankNet以pairwise error的方式计算cost，左图的cost为13，右图通过把第一个相关文档下调3个位置，第二个文档上条5个位置，将cost降为11，但是像NDCG或者ERR等评价指标只关注top k个结果的排序，在优化过程中下调前面相关文档的位置不是我们想要得到的结果。图 1右图左边黑色的箭头表示RankNet下一轮的调序方向和强度，但我们真正需要的是右边红色箭头代表的方向和强度，即更关注靠前位置的相关文档的排序位置的提升。LambdaRank正是基于这个思想演化而来，其中**Lambda指的就是红色箭头，代表下一次迭代优化的方向和强度，也就是梯度。**

**LambdaRank是一个经验算法，它不是通过显示定义损失函数再求梯度的方式对排序问题进行求解，而是分析排序问题需要的梯度的物理意义，直接定义梯度，即Lambda梯度。**

LambdaRank在RankNet的加速算法形式($\lambda_{ij}=\frac{\partial C_{ij}}{\partial s_i} = \sigma(\frac{1}{2}(1-S_{ij})-\frac{1}{1+e^{\sigma (s_i-s_j)}}),S_{ij}=1$)的基础上引入评价指标Z（如NDCG、ERR等），把交换两个文档的位置引起的评价指标的变化$|\Delta_{NDCG}|$作为其中一个因子，实验表明对模型效果有显著的提升：

$$\lambda_{ij}=\frac{\partial C(s_i-s_j)}{\partial s_i}=\frac{-\sigma}{1+e^{\sigma (s_i-s_j)}}|\Delta_{NDCG}|$$

损失函数的梯度代表了文档下一次迭代优化的方向和强度，由于引入了IR评价指标，Lambda梯度更关注位置靠前的优质文档的排序位置的提升。有效的避免了下调位置靠前优质文档的位置这种情况的发生。LambdaRank相比RankNet的优势在于分解因式后训练速度变快，同时考虑了评价指标，直接对问题求解，效果更明显。

## 2.4 LambdaMart
>1）Mart定义了一个框架，缺少一个梯度。
2）LambdaRank重新定义了梯度，赋予了梯度新的物理意义。

因此，所有可以使用梯度下降法求解的模型都可以使用这个梯度，MART就是其中一种，将梯度Lambda和MART结合就是大名鼎鼎的LambdaMART。

MART的原理是直接在函数空间对函数进行求解，模型结果由许多棵树组成，每棵树的拟合目标是损失函数的梯度，在LambdaMART中就是Lambda。LambdaMART的具体算法过程如下：

![lambdamart.jpg](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/notes/ltr-6.png)

**可以看出LambdaMART的框架其实就是MART，主要的创新在于中间计算的梯度使用的是Lambda，是pairwise的。MART需要设置的参数包括：树的数量M、叶子节点数L和学习率v，这3个参数可以通过验证集调节获取最优参数。**

MART支持“热启动”，即可以在已经训练好的模型基础上继续训练，在刚开始的时候通过初始化加载进来即可。

**下面简单介绍LambdaMART每一步的工作：**
1)  每棵树的训练会先遍历所有的训练数据（label不同的文档pair），计算每个pair互换位置导致的指标变化$|\Delta Z_{ij}|$以及Lambda，即$\lambda_{ij}=-\frac{1}{1+e^{s_i-s_j}}|\Delta Z_{ij}|$ ，然后计算每个文档的Lambda： $\lambda_i=\sum_{j:(i,j)\in I}\lambda_{ij}-\sum_{k:(k,i)\in I}\lambda_{ki}$，再计算每个$\lambda_i$ 的导数$w_i$，用于后面的Newton step求解叶子节点的数值。

2)  创建回归树拟合第一步生成的$\lambda_i$，划分树节点的标准是Mean Square Error，生成一颗叶子节点数为L的回归树。

3)  对第二步生成的回归树，计算每个叶子节点的数值，采用Newton step求解，即对落入该叶子节点的文档集，用公式$\frac{\sum_{x_i \in R_{lm}}y_i}{\sum_{x_i \in R_{lm}}w_i}$计算该叶子节点的输出值。

4)  更新模型，将当前学习到的回归树加入到已有的模型中，用学习率v（也叫shrinkage系数）做regularization。

**LambdaMART具有很多优势：**
1)  适用于排序场景：不是传统的通过分类或者回归的方法求解排序问题，而是直接求解

2)  损失函数可导：通过损失函数的转换，将类似于NDCG这种无法求导的IR评价指标转换成可以求导的函数，并且赋予了梯度的实际物理意义，数学解释非常漂亮

3)  增量学习：由于每次训练可以在已有的模型上继续训练，因此适合于增量学习

4)  组合特征：因为采用树模型，因此可以学到不同特征组合情况

5)  特征选择：因为是基于MART模型，因此也具有MART的优势，可以学到每个特征的重要性，可以做特征选择

6)  适用于正负样本比例失衡的数据：因为模型的训练对象具有不同label的文档pair，而不是预测每个文档的label，因此对正负样本比例失衡不敏感

[参考博文：Learning to Rank简介](https://www.cnblogs.com/bentuwuying/p/6681943.html)
[参考博文：Learning to Rank算法介绍：RankNet，LambdaRank，LambdaMart](https://www.cnblogs.com/bentuwuying/p/6690836.html)


---