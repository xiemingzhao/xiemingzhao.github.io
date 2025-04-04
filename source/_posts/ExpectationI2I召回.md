---
title: ExpectationI2I 召回
categories:
  - 召回模型
  - 算法总结
  
tags:
  - 召回
  - ExpectationI2I
  
mathjax: true
copyright: true
abbrlink: expectationi2irecall
date: 2021-11-28

---

## 1 引言

在推荐的发展历史中，Amazon 在 ItemCF 上进行了不少的探索。2003年，其在 IEEE INTERNET COMPUTING 发表的[《http://Amazon.com Recommendations: Item-to-Item Collaborative Filtering》](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)一文中提出了 ItemCF 推荐算法，引起了不小的波澜。其`主要优势是`：
* 简单可扩展；
* 可解释性强；
* 实时性高；

在早期，ItemCF/UserCF 往往被用于推荐主算法，在当下一般作为召回算法。**UserCF 适用于用户数的变化频率小于物品数的变化频率的场景，ItemCF 则相反。当今的互联网环境下确实是更适合 ItemCF 发挥。**


而本文就是为了介绍其新提出的一种改进的 ItemCF 算法 `ExpectationI2I`，当然有的地方名字可能不一样。这是由 Amazon 在 2017 年的 IEEE INTERNET COMPUTING 上发表的文章[《Two Decades of Recommender Systems at Amazon.com》](https://assets.amazon.science/76/9e/7eac89c14a838746e91dde0a5e9f/two-decades-of-recommender-systems-at-amazon.pdf)中介绍的。

<!--more-->

## 2 动机
主要是在如何定义物品的相关性上，有一定的改进空间。对于两个物品X和Y，在购买了X的用户中，假设有$N_{xy}$购买了Y。那在这里面，实际上有两部分组成：
* 一个是因为X和Y`相关`，而产生的`关联购物`；
* 另一个则是X和Y`不相关`，仅仅是`随机性导致的共同购物`。

所以，我们**核心目标就是要度量其中关联购物的那一部分**。在2003及之前的文章中，其大多使用下面的方法：

>首先假设购买X的user有同概率P(Y)购买Y，且 **P(Y)=购买Y的uv/所有购买的uv** 。那么X和Y的共同购买人数 $N_{xy}$ 的期望$E_{xy}$ = 购买X的uv * P(Y)。

这里在理解上，个人认为**有两点需要注意**：
1. $N_{xy}$ 实际上可以在实际中观测到，也**就是X和Y的共同购买uv数，但是包含随机性共同购物**；
2. 上面实际上不仅假设了购买同概率，还同时默认一个假设：**购买了X的用户群购买Y的概率与全局分布一致**。

而当出现很多购买两很大的用户时，就容易增加随机性共同购物，从而拉高了预估的结果。故，本文**最核心是：如何度量$N_{xy}$中因随机性而产生的共同购物次数**。那么剔除这一部分就可以得到因X和Y相关性而产生的`关联购物`次数，这便可以更有效的度量物品之间的相关性。

## 3 算法原理
为了计算购买X的用户会因随机性购买Y的期望$E_{xy}$，我们记：

* 对于任意购买了X的用户c，|c|=其购买总次数-购买了X的次数；
* P(Y)=全部用户购买Y的次数/全部用户的全部购买次数；
* 那么对于该用户c，在其除X外的|c|次购买中，有购买 Y 的概率$P_{xy} = 1-(1-P(Y))^{|c|}$

所以$E_{xy}$就可以通过对所有购买了X的用户的$P_{xy}$求和来得到:
$$
\begin{array}{l}
E_{XY} &=& \sum_{c \in X} [1 - (1-P(Y))^{|c|}] = \sum_{c \in X} [1 - \sum_{k=0}^{|c|} C_{|c|}^{k} (-P_Y)^k] \\
&=& \sum_{c \in X}[1 - [1 + \sum_{k=1}^{|c|} C_{|c|}^{k} (-P_Y^k)]] = \sum_{c \in X} \sum_{k=1}^{|c|} (-1)^{k+1} C_{|c|}^{k}P_Y^k \\
&=& \sum_{c \in X} \sum_{k=1}^{\infty} (-1)^{k+1} C_{|c|}^{k}P_Y^k \quad (since \ C_{|c|}^{k} = 0 \ for \ k>|c| ) \\
&=& \sum_{k=1}^{\infty} \sum_{c \in X} (-1)^{k+1} C_{|c|}^{k}P_Y^k \quad (Fubini's \ theorem) \\
&=& \sum_{k=1}^{\infty} \alpha_{k}(X) P_Y^k \quad where \ \alpha_k(X) = \sum_{c \in X}(-1)^{k+1} C_{|c|}^{k}
\end{array}
$$

作者提到一些**计算技巧**：
* 即实际中$P_Y$一般比较小，所以可以设置一个k的上限 `max_k` 作为求和多项式的最多项，也即做了**级数求和的近似**。
* 另一方面，为了降低复杂度，$P_Y$和$\alpha_{k}{X}$的各组合项可以提前计算好，降低阶乘重复计算的复杂度。

到这里，我们已经得到了购买X的用户随机购买Y的一个估计$E_{xy}$，据此可以判断真实中观测到的$N_{xy}$与随机估计的高低。所以，我们如果用同时购买的uv去除随机性的共同购买次数，得到的便是一个X和Y的关联性购买次数估计值。此便可以作为度量与X相关的商品Y的相关度，实际上作者还认为实际应用中有下面三种估计公式可选：
1. $N_{xy}-E_{xy}$，**会偏向于更流行的Y**;
2. $(N_{xy}-E_{xy})/E_{xy}$，**会使得低销量的物品很容易获得高分数**;
3. $(N_{xy}-E_{xy})/sqrt(E_{xy})$，**在高曝光商品和低销量商品之间找到平衡，需要动态调整**。


## 4 实际应用
在实际应用中，往往没有那么简单。首先我们介绍一下，该算法如何基于实际用户行为来计算，然后笔者将抛出一些问题和自己的优化建议。

$$\sum_{c \in X} \sum_{k=1}^{|c|} (-1)^{k+1} C_{|c|}^{k}P_Y^k$$

现实一般基于上述公式来计算比较方便，原因是先处理好每个 `user x item` 的统计量，然后按照 user 聚合即可。

`step1`
构建所有行为（点击收藏等）的表，保留行为数量在合理区间 [n1,n2] 内的 user，保留 session 数量 >m1 的 item_id
t1: user_id,item_id,time,session_id

`step2`
基于t1自join生成记录 session 内行为 pair 对表
t2: user_id,session_id,left_item_id,left_time,right_item_id,right_time,time_delta

`step3`
基于t2统计全局 pair 对数,session 去重 pair 对数统计表,计算 $N_{xy}$，
t3: left_item_id,right_item_id,cnt,user_session_cnt

`step4`
基于 expect_all_user_action 统计 user_id,item_id 共现的 session_id 次数
t4: user_id,item_id,sessioncnt

`step5:`
这一步最重要，**我们需要计算 user x item 的|c|，下面记为 parm_c**。记录 user_id,item_id 的统计参数
t5: user_id,item_id,clk_pv,clk_pv_all,parm_c(clk_pv_all-clk_pv)

**为了降低计算复杂度，这里的 parm_c 可以做上限截断**。有了 parm_c 后就可以计算 $\sum_{k=1}^{|c|} (-1)^{k+1} C_{|c|}^{k}P_Y^k$。

`step6:`
基于t5对 user 进行聚合，统计 item_id 维度的参数
t6: item_id,clk_cnt,clk_all,clk_prob,parm_c_list
**其中 clk_prob 便是$P_Y$，即一个物品在全局被随机点击的概率。parm_c_list 则是将物品在所有的 user 上的 parm_c汇总到一起，为了后面来计算 $E_{xy}$。**

`step7:`
基于t6构建 item pair，来拼接 $N_{xy}$，并计算最重要的 $E_{xy}$
t7: left_item_id,right_item_id,Nxy,left_parm_c_list,right_clk_prob,Exy
其中，$E_{xy}$ 可以通过 left_parm_c_list,right_clk_prob 来计算。例如：
```python
from scipy.special import comb
class ExpectScore(object):
    def evaluate(self, parm_c_list, clk_prob):

        def get_exy(parm_c, clk_prob):
            ans = 0
            for i in range(parm_c):
                ans += pow(-1, (k + 1)%2) * pow(prob_num,i) * comb(parm_c, i)
                if s_tmp[str(i)+'.0'] == 0:
                    break
            return sum(list(res_t.values()))
            
        try:
            res = 0
            for parm_c in parm_c_list.split(','):
                res += get_exy(parm_c, clk_prob)
            return sum(res)
        except Exception as e:
            return 0.0000
```

`step8:`
已经拿到了最重要的两个统计项$N_{xy},E_{xy}$，计算最终结果分，并对每个 left_item_id 的所有 right_item_id 降序排列截断构建倒排即可。
t8: left_item_id,right_item_id, sc, rank
其中，
* sc 就是前述融合分数公式来计算得到的;
* rank 便是按照sc降序排列得到的排名。

实际上，如果计算中由于 parm_c 的上限截取的比较大，那么在计算$E_{xy}$中会频繁的遇到较大值的排列组合$C_{|c|}^{k}$的计算，有可能速度会比较慢。而在全局|c|的上限固定的情况下，我们可以利用**空间换时间来优化**这个：
>* 因为|c|的上限固定，会使得$C_{|c|}^{k}$的计算比较高频重复，所以可以提前遍历全部有效k(即k<=｜c|)的$C_{|c|}^{k}$结果；
>* 然后，将表t5中的 parm_c 替换成存储各拆分项$C_{|c|}^{k}$的结果，在表t6中的 parm_c_list 也类似存储各个 user 的 parm_c 对应的各个拆分项；
>* 最后，在表t7的$E_{xy}$计算环节，就不用计算排列组合项`$C_{|c|}^{k}$`，只需要将存好的`comb(parm_c, i)`项带入进行计算即可。


## 5 问题与思考
实际上，如前文所述，这种算法不喜欢 user 行为覆盖过多的物品，**高活用户对其不友好**。而像 item2vec 或者 wbcos 这种对此就不敏感，因为基于 user 行为序列内构建样本 pair 对的。

在前面，介绍了 `ExpectationI2I` 算法的原理，实际构建方式，以及优化计算复杂度的方案。然而，实际应用中往往还存不少 badcase，我们需要对一些环节做一些精细的优化，这里列出部分问题和方案的思考，供备忘和参考。

**1. 有效行为数据的筛选逻辑**
部分行为过少或过多的极端用户会影响算法稳定性，所以在应用中我们也提到了n1,n2以及m1等边界超参数。这些参数的具体选值需要根据具体场景数据分布来确定，整体目标就是剔除不稳健的用户行为数据。

**2. 类目的信息，例如不同类目的惩罚或者加权；**
在构建的pair对上计算各统计项的时候，需要考虑类目相关度的一个影响，比如同类目是否进行加权，异类目是否进行降权，或者直接根据类目分布来进行权重调整，目的就是为了降低行为中偶然性类目的搭便车情况，降低算法的badcase率。

**3. 融合分公式的调整；**
在算法原理模块，已经介绍了论文作者提出的三种公式及其特性。在实际应用中，往往还需要进行一定的调整，否则很容易出现一些badcase。例如一种改进公式$\frac{N_{xy}-E_{xy}}{1 + \sqrt{E_{xy}}}$，总之还是以实际线上实验数据为准。

**4. 天级别数据的增量融合；**
在应用次算法的时候，往往面临一个行为数据时间窗口的选取，一次性选择太多，计算量会过大，选择太少数据会不准确。那么，一般可以采用每日增量更新。有两种方式：
* 增量更新$N_{xy},E_{xy}$两个统计项，分别赋予历史值和增量值合适的权重；
* 增量直接更新表t8中最终的分数sc，然后构建新的排序。


**参考文献**
[【翻译+批注】亚马逊推荐二十年](https://zhuanlan.zhihu.com/p/27656480)
[【论文阅读】Two Decades of Recommender Systems at Amazon.com](https://www.jianshu.com/p/248209fd1038)
[Two Decades of Recommender Systems at Amazon.com](https://assets.amazon.science/76/9e/7eac89c14a838746e91dde0a5e9f/two-decades-of-recommender-systems-at-amazon.pdf)

---