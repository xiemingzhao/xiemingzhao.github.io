---
title: LightGBM A Highly Efficient Gradient Boosting Decision Tree （论文解析）
categories:
  - 学习笔记
  - 论文解析
tags:
  - 机器学习
  - LightGBM
mathjax: true
copyright: true
abbrlink: lightgbmpaper
date: 2019-06-23

---

[原始论文：LightGBM-A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)

# LightGBM 一种高效的梯度提升决策树

## 摘要
`Gradient Boosting Decision Tree (GBDT)`是一个非常流行的机器学习算法，却只有像XGBoost和pGBRT的一些实现。尽管许多工程上的优化方案已经在这些实现中应用了，但是当特征维度较高和数据量巨大的时候，仍然存在效率和可扩展性的问题。一个主要原因就是对于每一个特征的每一个分裂点，都需要遍历全部数据计算信息增益，这一过程非常耗时。针对这一问题，本文提出两种新方法：Gradient-based One-Side Sampling (GOSS) 和Exclusive Feature Bundling (EFB)（基于梯度的one-side采样和互斥的特征捆绑）。在GOSS中，我们排除了一部分重要的具有小梯度实例数据的比例，只用剩下的来估计信息增益。我们证明，这些梯度大的实例在计算信息增益中扮演重要角色，GOSS可以用更小的数据量对信息增益进行相当准确的估计。对于EFB，我们捆绑互斥的特征（例如，特征间很少同时非零的特征），来降低特征的个数。我们完美地证明了捆绑互斥特征是NP难的，但贪心算法能够实现相当好的逼近率，因此我们能够在不损害分割点准确率许多的情况下，有效减少特征的数量。（牺牲一点分割准确率降低特征数量），这一算法命名为LightGBM。我们在多个公共数据集实验证明，LightGBM加速了传统GBDT训练过程20倍以上，同时达到了几乎相同的精度。

<!--more-->

## 1 引言
GBDT是一个广泛应用地机器学习算法，这得益于其本身的有效性、准确性、可解释性。GBDT在许多机器学习任务上均取得了最好的效果，例如多分类，点击预测，排序。但最近几年随着大数据的爆发（特征量和数据量），GBDT正在面临新的挑战，特别是在平衡准确率和效率的调整方面。常见的GBDT的实现，对于每个特征,都需要遍历全部数据来计算所有可能分裂点的信息增益。因此，其计算复杂度将受到特征数量和数据量双重影响，造成处理大数据时十分耗时。

为了解决这一问题，一个直接的方法就是减少特征量和数据量而且不影响精确度。然而，这将是非常重要的。例如，我们不清楚如何针对提升GBDT来进行数据抽样。而有部分工作根据数据权重采样来加速boosting的过程，它们不能直接地应用于GBDT，因为gbdt没有样本权重。在本文中，我们提出两种新方法实现此目标。

Gradient-based One-Side Sampling (GOSS)。尽管GBDT虽然没有数据实例权重，但每个数据实例有不同的梯度，从而在信息增益的计算中扮演不同的角色。特别地，根据计算信息增益的定义，梯度大的实例对信息增益有更大的影响。因此，在数据实例下采样时，为了保持信息增益预估的准确性，我们应该尽量保留梯度大的样本（预先设定阈值，或者最高百分位间），并且随机去掉梯度小的样本。我们证明此措施在相同的采样率下比随机采样获得更准确的结果，尤其是在信息增益范围较大时。

Exclusive Feature Bundling (EFB)。通常在真实应用中，虽然特征量比较多，但是由于特征空间十分稀疏，那我们是否可以设计一种无损的方法来减少有效特征呢？特别在，稀疏特征空间上，许多特征几乎都是互斥的（例如像文本挖掘中的one-hot特征）。我们就可以捆绑这些互斥的特征。最后，我们设计了一个有效的算法，将捆绑问题简化成图着色问题（方法是将特征作为节点，在每两个不完全互斥的特征之间添加边），并且通过贪心算法可以求得近似解。

我们将这种结合了 GOSS 和 EFB 的新 GBDT 算法称为*LightGBM*。我们在多个公开数据集上的实验结果证明了 LightGBM 在得到几乎相同准确率的情况下能够提升20倍的训练速度。

这篇文章剩下的部分将按如下安排。首先，我们在第二部分回顾了 GBDT 算法和相关工作。然后，我们分别在第三和第四部分介绍了 GOSS 和 EFB 的详细内容。在第五部分，展示了我们在公共数据集上所做的关于 LightGBM 的实验结果。最后，我们在第六部分进行了总结。

## 2 预研
### 2.1 GBDT 和它的复杂度分析
GBDT是一种集成模型的决策树，顺序训练决策树。每次迭代中，GBDT通过拟合负梯度（也被称为残差）来学到决策树。

学习决策树是GBDT主要的时间花销，而学习决策树中找到最优切分点最消耗时间。有一种最常用的预排序算法来找到最优切分点，这种方法会列举预排序中所有可能的切分点。这种算法虽然能够找到最优的切分点，但在训练速度和内存消耗上的效率都很低。另一种流行算法是直方图算法（histogram-based algorithm），如 Alg.1 所示。直方图算法并不通过特征排序找到最优的切分点，而是将连续的特征值抽象成离散的分箱，并使用这些分箱在训练过程中构建特征直方图。这种算法更加训练速度和内存消耗上都更加高效，lightGBM使用此种算法。

histogram-based算法通过直方图寻找最优切分点，其建直方图消耗O(#data * #feature)，寻找最优切分点消耗O(#bin * # feature)，而#bin的数量远小于#data，所以建直方图为主要时间消耗。如果能够减少数据量或特征量，那么还能够够加速GBDT的训练。（寻找最优切分点已经进行了优化，那么我们现在应该对建直方图的时间进行优化）

## 2.2 相关工作
GBDT有许多实现，如XGBoost，PGBRT，Scikit-learn，gbm in R。Scikit-learn和gbm in R实现都用了预排序，pGBRT使用了直方图算法。XGBoost支持预排序和直方图算法，由于XGBoost胜过其他算法，我们用它作为实验的baseline。

为了减小训练数据集，通常做法是下采样。例如过滤掉权重小于阈值的数据。SGB每次迭代中用随机子集训练弱学习器。或者采样率基于训练过程动态调整。然而，这些都是使用基于AdaBoost的SGB，其不能直接应用于GBDT是因为GBDT中没有原始的权重。虽然SGB也能间接应用于GBDT，但往往会影响精度。

同样，可以考虑过滤掉弱特征（什么是弱特征）来减少特征量。通常用主成分分析或者投影法。当然，这些方法依赖于一个假设-特征有高冗余性，但实际中往往不是。（设计特征来自于其独特的贡献，移除任何一维度都可以某种程度上影响精度）。

实际中大规模的数据集通常都是非常稀疏的，使用预排序算法的GBDT能够通过无视为0的特征来降低训练时间消耗。然而直方图算法没有优化稀疏的方案。因为直方图算法无论特征值是否为0，都需要为每个数据检索特征区间值。如果基于直方图的GBDT能够有效解决稀疏特征中的0值，并且这样将会有很好的性能。

为了解决前面工作的局限性，我们提出了两个全新的技术分别是 Gradient-based One-Side Sampling (GOSS) 和 Exclusive Feature Bundling (EFB)（基于梯度的one-side采样和互斥的特征捆绑）。跟多的细节会再下一部分介绍。

![Alg.1 & Alg.2](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/lgbm1.JPG)

## 3 基于梯度的one-side采样
在这一部分，我们为 GBDT 提出了一个新的抽样方法， 这能够在减少数据实例个数和保持学习到的决策树的准确度之间达到一个平衡。

### 3.1 算法描述
在AdaBoost中，样本权重是数据实例重要性的指标。然而在GBDT中没有原始样本权重，不能应用权重采样。幸运的事，我们观察到GBDT中每个数据都有不同的梯度值，对采样十分有用，即实例的梯度小，实例训练误差也就较小，已经被学习得很好了，直接想法就是丢掉这部分梯度小的数据。然而这样做会改变数据的分布，将会影响训练的模型的精确度，为了避免此问题，我们提出了GOSS。

GOSS保留所有的梯度较大的实例，在梯度小的实例上使用随机采样。为了抵消对数据分布的影响，计算信息增益的时候，GOSS对小梯度的数据引入常量乘数。GOSS首先根据数据的梯度绝对值排序，选取top a x 100%个实例。然后在剩余的数据中随机采样bx100%个实例。接着计算信息增益时为采样出的小梯度数据乘以(1-a)/b（即，小梯度样本总数/随机采样出的小梯度样本数量），这样算法就会更关注训练不足的实例，而不会过多改变原数据集的分布。

### 3.2 理论分析
GBDT使用决策树，来学习获得一个将输入空间$\mathcal \chi^s$映射到梯度空间$\mathcal G$的函数。假设训练集有n个实例 ${x_1,...,x_n}$，每个$x_i$都是一个维度为s的特征向量。每次迭代时，模型数据变量的损失函数的负梯度方向表示为$g_1,...,g_n$，决策树通过最优切分点（最大信息增益点）将数据分到各个节点。对于GBDT，一般通过分割后的方差衡量信息增益，具体由下定义。

**定义3.1**：$O$表示某个固定叶子节点的训练集，分割特征j的分割点d对应的方差增益定义为：

$$V_{j|O} (d) = \frac{1}{n_O} ( \frac{(\sum_{ \{x_i \in O:x_{ij} \leq d\} } g_i)^2} {n_{l|O}^j (d)} + \frac{(\sum_{ \{x_i \in O:x_{ij} > d\} } g_i)^2} {n_{r|O}^j (d)} )$$

其中$n_O = \sum I[x_i \in O]$(*某个固定叶子节点的训练集样本的个数*)，$n_{l|O}^j (d) = \sum I[x_i \in O:x_{ij} \leq d]$（*在第j个特征上值小于等于d的样本个数*），和 $n_{r|O}^j (d) = \sum I[x_i \in O:x_{ij} > d]$（*在第j个特征上值大于d的样本个数*）。

对于特征 j，决策树算法选择$d_j^* = argmax_d V_j(d)$并且计算最大信息增益$V_j(d_j^*)$。然后，数据集会根据特征$j^*$在点$d_j^*$分到左右子节点中去。

在我们所提出的GOSS方法中，首先，我们训练实例按照它们梯度的绝对值进行降序排列；第二，我们保留梯度最大的top-a x 100%个实例作为样本子集A；再者，对于剩下的包含(1-a) x 100%个更小梯度实例的子集$A^c$，我们进一步随机抽样一个大小为$b x |A^c|$的子集B；最后，我们我们将样本实例按照下列公式在子集$A \cup B$上的方法增益估计值进行分割：

$$\tilde V_j(d) = \frac{1}{n} (\frac{(\sum_{x_i \in A_l} g_i + \frac{1-a}{b} \sum_{x_i \in B_l} g_i)^2}{n_l^j (d)} + \frac{(\sum_{x_i \in A_r} g_i + \frac{1-a}{b} \sum_{x_i \in B_r} g_i)^2}{n_r^j (d)}),     (1)$$

其中，$A_l = {x_i \in A:x_{ij} \leq A}, A_r = {x_i \in A:x_{ij} > d}, B_l = {x_i \in b:x_{ij} \leq d}, B_r = {x_i \in B:x_{ij} > d}$，并且系数(1-a)/b是用来将B上的梯度和归一化到$A^c$的大小上去。

因此，在GOSS中，我们使用更小实例子集上的估计值$\tilde V_j (d)$而不是使用所有的实例来计算精确的$V_j (d)$来得到分裂点，并且这种计算成本也可以得到大大地降低。更重要的是，下列定理表名了GOSS不会损失更多的训练精度并且会优于随机抽样。由于空间限制，我们将定理的证明放在了补充材料中。

**定理3.2** 我们将GOSS的近似误差定义为$\varepsilon (d) = |\tilde V_j (d) - V_j (d)| \ and\ \bar g_l^j (d) = \frac{\sum_{x_i \in (A \cup A^c)_l |g_i|} }{n_l^j (d)}, \bar g_r^j (d)  = \frac{\sum_{x_i \in (A \cup A^c)_r |g_i|} }{n_r^j (d)}$。概率至少是$1- \delta$，我们有：

$$\varepsilon (d) \leq C_{a,b}^2 ln 1/\delta \cdot max\{ \frac{1}{n_l^j(d)}, \frac{1}{n_r^j(d)} \} + 2DC_{a,b} \sqrt{\frac{ln 1/\delta}{n} },   (2)$$

其中$C_{a,b} = \frac{1-a}{\sqrt{b}} max_{x_i \in A^c} |g_i|$， 和 $D = max(\bar g_l^j (d), \bar g_r^j (d) )$。

根据定理，我们可以得到以下结论：(1)GOSS的渐进近似比率是$\mathcal O(\frac{1}{n_l^j (d)} + \frac{1}{n_r^j (d)} + \frac{1}{\sqrt{n}} )$。如果分割的不是特别不平衡(即$n_l^h \geq \mathcal O (\sqrt n)$ 且 $n_r^h \geq \mathcal O (\sqrt n)$)，近似误差可以由公式(2)中的第二项来表示，其中当$n \rightarrow \infty$时$\mathcal O (\sqrt n)$将趋向于0。这意味着当数据集个数很大的时候，近似值将是很准确的。(2)随机抽样是一个$a = 0$时GOSS的特例。在许多案例中，GOSS都会表现地比随机抽样要好，在$C_{0,\beta} > C_{a,\beta - a}$条件下，这等价于$\frac{\alpha_a}{\sqrt \beta} > \frac{1-a}{\sqrt{\beta - a} }$且有$\alpha_a = max_{x_i \in A\cup A^c} |g_i|/max_{x_i \in A^c |g_i|}$。

下一步，我们分析了GOSS的泛化表现。我们考虑了GOSS的泛化误差$\varepsilon_{gen}^{GOSS} (d) = |\tilde V_j (d) - V_* (d)|$，这个值是由GOSS中抽样训练得到的方差增益和潜在分布的真实方差增益之间的差值。我们有$\varepsilon_{gen}^{GOSS} (d) \leq |\tilde V_j (d) - V_j (d)| + |V_j (d) - V_* (d)| \triangleq \varepsilon_{GOSS} (d) + \varepsilon_{gen} (d)$。因此，如果GOSS近似是准确的，那么带GOSS的泛化误差是接近于使用全量数据集计算得结果。另一方面，抽样会增加基础学习器之间的多样性，这潜在地帮助提升了泛化的表现。

## 4 互斥特征捆绑
这一章，我们提出了一个全新的方法来有效地减少特征数量。

![Alg.3 & Alg.4](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/lgbm2.JPG)

高维的数据通常是非常稀疏的。这种稀疏性启发我们可以设计一种无损地方法来减少特征的维度。特别地，在稀疏特征空间中，许多特征是完全互斥的，即它们从不同时为非零值。我们可以绑定互斥的特征为单一特征（这就是我们所说的互斥特征捆绑）。通过仔细设计特征扫描算法，我们从特征捆绑中构建了与单个特征相同的特征直方图。这种方式的构建直方图时间复杂度从O(#data * #feature)降到O(#data * #bundle)，由于#bundle << # feature，我们能够极大地加速GBDT的训练过程而且不损失精度。(构造直方图的时候，遍历一个“捆绑的大特征”可以得到一组exclusive feature的直方图。这样只需要遍历这些“大特征”就可以获取到所有特征的直方图，降低了需要遍历的特征量。)在下面，我们将会展示如何实现这些方法的细节。

有两个问题需要被解决。第一个就是需要确定哪些特征后应该绑定在一起。第二个就是如何构造捆绑。

**定理4.1** *将特征分割为较小量的互斥特征群是NP难的。*

*证明：*将图着色问题归约为此问题。而图着色是NP难的，所以我们可以得到我们的结论。

给定图着色的一个实例G=(V, E)。可以构建一个我们问题的一个实例如下所示。以G的关联矩阵的每一行为特征，得到我们问题的一个实例有|V|个特征。 很容易看到，在我们的问题中，一个独特的特征捆绑与一组具有相同颜色的顶点相对应，反之亦然。

对于第1个问题，我们在定理4.1说明寻找一个最优的捆绑策略是NP难的，这就表明不可能找到一个能够在多项式时间内解决的办法。为了寻找好的近似算法，我们将最优捆绑问题归结为图着色问题，如果两个特征之间不是相互排斥，那么我们用一个边将他们连接，然后用合理的贪婪算法（具有恒定的近似比）用于图着色来做特征捆绑。 此外，我们注意到通常有很多特征，尽管不是100％相互排斥的，也很少同时取非零值。 如果我们的算法可以允许一小部分的冲突，我们可以得到更少的特征包，进一步提高计算效率。经过简单的计算，随机污染小部分特征值将影响精度最多为$\mathcal O([(1-\gamma) n]^{-2/3})$(参考文献【2】)，$\gamma$是每个绑定中的最大冲突比率。所以，如果我们能选择一个相对较小的$\gamma$时，能够完成精度和效率之间的平衡。

基于上述的讨论，我们针对互斥特征捆绑设计了一个算法如Alg.3所示。首先，我们建立一个图，每个点代表特征，每个边有权重，其权重和特征之间总体冲突相关。第二，我们按照降序排列图中点的度来排序特征。最后，我们检查排序之后的每个特征，对它进行特征绑定或者建立新的绑定使得操作之后的总体冲突最小（由$\gamma$控制）。算法3的时间复杂度是$\mathcal O (\# feature ^2)$，并且只在训练之前处理一次。其时间复杂度在特征不是特别多的情况下是可以接受的，但难以应对百万维的特征。为了继续提高效率，我们提出了一个更加高效的不用构建图的排序策略：将特征按照非零值个数排序，这和使用图节点的度排序相似，因为更多的非零值通常会导致冲突。新算法在算法3基础上只是改变了排序策略来避免重复。

对于第2个问题，我们需要一个好的办法合并同一个bundle的特征来降低训练时间复杂度。关键在于原始特征值可以从bundle中区分出来。鉴于直方图算法存储离散值而不是连续特征值，我们通过将互斥特征放在不同的箱中来构建bundle。这可以通过将偏移量添加到特征原始值中实现，例如，假设bundle中有两个特征，原始特征A取值[0, 10]，B取值[0, 20]。我们添加偏移量10到B中，因此B取值[10, 30]。通过这种做法，就可以安全地将A、B特征合并，使用一个取值[0, 30]的特征取代A和B。算法见Alg.4。

EFB算法能够将许多互斥的特征变为低维稠密的特征，就能够有效的避免不必要0值特征的计算。实际，对每一个特征，建立一个记录数据中的非零值的表，通过用这个表，来忽略零值特征，达到优化基础的直方图算法的目的。通过扫描表中的数据，建直方图的时间复杂度将从O(#data)降到O(#non_zero_data)。当然，这种方法在构建树过程中需要而额外的内存和计算开销来维持这种表。我们在lightGBM中将此优化作为基本函数.因为当bundles是稀疏的时候，这个优化与EFB不冲突（可以用于EFB）.

## 5 实验
在这一部分，我们汇报了我们提出的LightGBM算法的实验结果。我们使用五个不同的公开数据集。这些数据集的细节列在了表1中。在它们中，微软的排序数据集包含30K网站搜索请求数据。这个数据集中的特征大多是稠密数值特征。Allstate
Insurance Claim和Flight Delay数据集都包含许多one-hot编码特征。并且最后两个数据集来自KDD CUP 2010 and KDD CUP 2012。我们直接地使用这些由获胜者NTU提供的特征，其中包含稠密特征和稀疏特征，并且这两个数据集非常大。这些数据集都是很大的，同时包含稀疏特征和稠密特征，并且涵盖了许多真实的任务。因此，我们直接地可以使用它们来测试我们的算法。

我们的实验环境是一个Linux服务器，包含两个E5-2670 v3 CPUs（总共24核）和256GB的内存。所有试验都是多线程运行并且线程的个数固定在16。

### 5.1 全部对比
![table of experiment](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/lgbm3.JPG)

我们在这一部分展示了所有的实验对比。XGBoost和不包含GOSS以及EFB（称为lgb_baseline）的LightGBM用作基准线。对于XGBoost，我们使用两个版本，xgb_exa(预排序算法)和xgb_his(基于直方图的算法)。对于xgb_his，lgb_baseline，和LightGBM，我们使用leaf-wise树增长方法。对于xgb_exa，因为它仅仅支持layer-wise增长策略，我们将xgb_exa的参数设成使其和其他方法增长相似的树。我们也可以通过调整参数使其在所有的数据集上能在速度和准确率上面达到平衡。我们在Allstate, KDD10 和 KDD12上设定a=0.05,b=0.05，并且在Flight Delay 和 LETOR上设定a = 0.1; b = 0.1。我们对于EFB设定$\gamma = 0$。所有的算法都运行固定的线程数，并且我们从迭代过程中获取最好的分数的准确率结果。

![training curves](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/lgbm4.JPG)

训练时间和测试准确率分别汇总在表2和表3中。从这些结果中，我们能够看到在于基准线保持几乎相同准确率的时候是最快速的。xgb_exa是基于预排序的算法，这相对于基于直方图的算法是非常慢的。相对于lgb_baseline，LightGBM在Allstate, Flight Delay, LETOR, KDD10 和 KDD12数据集上分别加速了21，6，1.6，14和13倍。因xgb_his非常消耗内存，导致其在KDD10 和 KDD12数据集上内存溢出而不能成功运行。在剩下的数据集上，LightGBM都是最快的，最高是在Allstate数据集上加速了9倍。由于所有的算法都在差不多的迭代次数后收敛了，所以加速是基于平均每次迭代时间计算得到的。为了展示整个训练过程，我们基于Flight Delay 和 LETOR的经过的时间也分别展示了训练曲线在图1和图2中。为了节省空间，我们将其他数据集的训练曲线放在了补充材料中。

![Accuracy comparison](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/lgbm5.JPG)

在所有的数据集上，LightGBM都能够得到几乎和基准线一致的测试准确率。这表明GOSS和EFB都不会降低准确率并且能够带来显著地加速。这也与我们前面的理论分析保持一致。

LightGBM在不同的数据集上得到了不同的加速率。整体的加速结果是来自GOSS和EFB二者的联合，我们将下一部分分开讨论二者的贡献。

### 5.2 GOSS的分析
首先，我们研究了GOSS的加速性能。从表2中LightGBM和EFB_only的对比来看，我们能够发现GOSS能够通过其自己在使用10%-20%的数据时候带来近2倍的加速。GOSS能够紧紧使用抽样数据进行学习树。然而，它依然保留了一些在全部数据集上的一些计算，例如预测和计算梯度。因此，我们能够发现整体的加速相对于抽样数据的百分比并不是线性的。然而，GOSS带来的加速又是非常显著地，并且这一技术可以普遍的应用在不同的数据集上。

第二，我们通过和SGB（随机梯度提升）对比评估了GOSS的准确率。为了没有泛化性的损失，我们使用LETOR数据集来进行测试。我们通过选择GOSS中不同的a和b值来设定抽样率，并且在SGB上使用同样的整体抽样率。我们使用这些设定运行并且使用early stopping直到其收敛。结果如表4所示。我们能够看到但我们使用相同的抽样率的时候GOSS的准确率总是比SGB要好。这一结果和我们3.2部分的讨论保持一致。所有的实验结果都表明了GOSS相比于随机抽样是一个更有效的抽样方法。

### 5.3 EFB的分析
我们通过对比lgb_baseline和EFB_only来检测了EFB在加速方面的贡献。结果如表2所示。这里我们没有允许捆绑发现流程中冲突的存在（即$\gamma = 0$）。我们发现EFB能够有助于在大规模数据集上获得显著性的加速。

请注意lgb_baseline已经在系数特征上进行了优化，且EFB依然能够在训练过程中进行加速。这是因为EFB将许多稀疏特征（one-hot编码的特征和一些潜在的互斥特征）合并成了很少的特征。基础的稀疏特征优化包含在了捆绑程序中。然而，EFB在树训练过程中为每个特征维持非零数据表上没有额外的成本。而且，由于许多预先独立出的特征捆绑到了一起，它能够增加本地空间并且能够显著地改善缓存冲击率。因此，在效率上的整体改进是显著地。基于上述分析，EFB是一个非常有效能够在基于直方图的算法中充分利用稀疏性的算法，并且它能够再GBDT训练过程中带来显著性加速。

## 6 结论
在这篇文章中，我们提出了全新的GBDT算法叫做LightGBM，它包含了连个新颖的技术：Gradient-based One-Side Sampling (GOSS) 和Exclusive Feature Bundling (EFB)（基于梯度的one-side采样和互斥的特征捆绑）分别来处理大数据量和高维特征的场景。我们在理论分析和实验研究表明，得益于GOSS和EFB，LightGBM在计算速度和内存消耗上明显优于XGBoost和SGB。未来工作中，我们将研究在GOSS中选择a，b值的优化方案，并且继续提高EFB在高维特征上的性能，无论其是否是稀疏的。

**参考文章**
[Lightgbm源论文解析-anshuai_aw1](https://blog.csdn.net/anshuai_aw1/article/details/83048709)

---