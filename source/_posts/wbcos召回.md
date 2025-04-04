---
title: wbcos 召回
categories:
  - 召回模型
  - 算法总结
  
tags:
  - 召回
  - wbcos
  
mathjax: true
copyright: true
abbrlink: wbcosrecall
date: 2022-05-09

---

## 1 背景
>`wb` 意为 `weight base`，wbcos 即加权式的 cos。

**思想：其实就是改进的 itemcos 来计算相似度。**

核心在于两点：
* user+session 内的 pair 重复出现的时候如何聚合，主要就是时间衰减和类目等维度加权；
* user+session 间的 pair 如何聚合，主要是 session 丰富度加权；

<!--more-->

## 2 步骤
### step0: 样本构造
将用户在 app 全场景的正行为汇总到一起，作为底表 `user_action_database`。

*注意可筛选行为数较多或者较少的，例如：正反馈item个数在[a,b]之间；以及高质量用户，例如经验：在近 n 天内有 order 的用户，以及用户当天点击数不少于 k 等等*

保留 `user,event,time,session,item,cate,brand` 等等维度。

### step1: 计算`bw`
在 `user+session` 维度下，计算：

$$userBw = \frac{1}{log_2 (3 + itemNum)}$$

其中 `itemNum` 指的是 user 在 session 内的正反馈 item 的去重个数。

**这里的思想很简单：即一个 user 在一个 session 维度下，看的 item 越多，理论上兴趣分布越广，则权重越小；从概率学角度理解集合元素越多，产生某 pair 对的概率越大，分得的权重也越小**

### step2: 计算 item 的 `wb`
在 user+session 维度下，计算同一 item 的出现次数 `itemCnt`，截断后作为 `itemWb`：

$$itemWb = min(m, itemCnt)$$

**注意，这里截断 m 是为了后续取 pair 对 topk 时间相近，思想就是：在找出与当前 itemA 行为最近的 itemB 的时候，后者有多次出现的话最多取 m 个（时间最近的）来构建 pair，m 具体需要根据业务数据来确定**

同时计算类目等维度的权重系数 `ratio`，这里以类目 cate 为例。

>即 `user_id,session_id` 维度下，每个 item_id 对应的类目权重。每个类目的权重可以参考：

$$ratio_k = \frac{cnt(cate_k)}{\sum_i cnt(cate_i)}$$

### step3: 计算 item pair 相关参数
在 `user+session` 维度下，构建 item 的 pair 对，可以设置 item 不同的时候才成 pair，于是每个 pair 就有两个 item，我们记为`(litem, ritem)`。

紧接着对每个pair对计算参数 `timeGap` 和 `matchRatio`。

$$timeGap = e^{- \alpha * abs(tsDiff)}$$

其中:
* $\alpha$ 是时间衰减的超参数，经验上可取 0.75。
* `tsDiff` 表示的是pair对中的两个正反馈 item 的行为发生时间差，建议使用 hour 的精度。

`matchRatio`的计算需要融入先验信息，我们以简单的cate维度为例：
```
matchRatio = 
case 
when 叶子类目相同 then 1
when 二级类目相同 then 0.9
when 一级类目相同 then 0.8
else 0.3
```
可以看得出来，此处是 pair 对中两个 item 的 cate 维度越相同，则先验相关性越高。此处，可以融入其他的先验信息，例如 brand，price 等。

### step4: 统计 pair 对的两种频次
首先，统计每一个pair对`(litem, ritem)`全局的频次，记为`pairCnt`，并且可以以此筛选除去总出现次数较少的 pair 对，例如`pairCnt>=5`。

其次，计算每一个pair对`(litem, ritem)`在全局有多少个 `user+session` 出现了，即以 `user+session` 为 key 去 groupby，来计算 `count(distinct user,session)`，我们记为 `pairUserSessionCnt`。

>*这里有些 tf-idf 的思想。*

### step5: 计算`innerProduct`参数。

以上三个参数的计算都是在`user+session+pair(litem, ritem)`维度之下的，我们记为`基准维度`。

#### 首先，计算`timeGapWeight`

我们知道在`基准维度`下，`matchRatio,pairCnt,pairUserSessionCnt`都是一致的，但是同一 pair 对会出现多次，每个 pair 对我们在前面计算过`timeGap`。而每个 pair 对`(litem,ritem)`都有自己的`itemWb`.

于是我们可以如下计算：
```
k = litemWb * ritemWb;
在基准维度下，对重复出现的 pair 对的 timeGap（记为timeGaps）取前k个，即：
timeGaps.sort(reverse = True);
最终，timeGapWeight = sum(timeGaps[:k])
```

#### 然后，我们计算`innerProduct`

这个就比较简单了，也是在去重后的`基准维度`上进行计算：

$$innerProduct = matchRatio * timeGapWeight * lratio * rratio $$

其中，`lratio` 和 `rratio` 分别是 step3 中计算的左右 item 的类目权重（可选）。

**这里其实可以理解为，同一 user+session 下，多次共现的 pair 对（matchRatio一致），按照其时间间隔权重来加权，两个 ratio 是按照对应类目集中度来加权（可选）**。

到这里，我们应该在`基准维度`（`user+session+pair(litem, ritem)`）下获得了以下特征数据(**此处已经去重了**)：

>matchRatio: 匹配度
timeGapWeight: 时间间隔权重
innerProduct: 内积权重
pairCnt: 全局pair的计数
pairUserSessionCnt: 出现对应pair的UserSession计数
litemWb: pair对中左item的wb
ritemWb: pair对右左item的wb
lratio: pair对中左item的类目权重ratio
rratio: pair对右左item的类目权重ratio
userBw: user+session级别的bw

### step6: 计算pair对的三个参数
首先，我们基于 step5 计算 `pairBw`：

$$pairBw = innerProduct * userBw^2$$

接着计算 pair 对中左右 item 的加权 wb（item 对应的 userBw）：

$$leftWb = litemWb * userBw$$
$$rightWb = ritemWb * userBw$$

### step7: 计算`itemWbLen`并聚合 pair 对
首先，我们之前通过截断`itemCnt`，作为`itemWb`，在这里我们不再需要`基准维度`，我们聚合到 pair 维度，以左 item 为 key 聚合出左右 item 的`itemWbLen`:

$$litemWbLen = \sqrt {sum(leftWb^2)}$$
$$ritemWbLen = \sqrt {sum(rightWb^2)}$$

其次，聚合pair对。
我们聚合全局的 pair 对，并计算下列参数：

$$pairBwScore = sum(pairBw)$$

### step8: 计算最终`wbScore`
最后我们将基于pair对维度计算：

$$wbScore = \frac{pairBwScore}{litemWbLen * ritemWbLen}$$

其中，`litemWbLen`和`ritemWbLen`分别是 pair 对中左右 item 的`itemWbLen`值。


## 3 后记
**1. 聚合**
在实际中应用的时候，往往每个分区生产一张 wbcos 分区结果表，我们可以进行多分区维度的**聚合**来减少方差从而提高准确度：
一般就是采用如下更新方式

$$wbScore = \frac{prePairBw + curPairBw}{(prelitemWbLen + curlitemWbLen) + (preritemWbLen + curritemWbLen)}$$

其实就是利用多窗口的数据进行指标平滑的思想。或者可以进行滑动平均，比如：

$$wbScore = \alpha * wbScore_{pre} + (1 - \alpha) * wbScore_{cur}$$

**2. 计算**
因为 i2i 召回逻辑上具有对称性，在构建 pair 时，只需要构建单向 pair 对 $(i1, i2)$ 即可。最终构建倒排时，反向 pair 对 $(i2, i1)$ 可以使用同样的相似度分，以减少计算量。

**3. 不同行为的融合**
在操作中，如何考虑所有的正行为，除`clk`之外，例如`fav`和`buy`等。那么对于不同行为之间的 pair 对就可以采取不一样的操作。主要是 session 内合并的时候所用的权重，在计算`innerProduct`的时候，`timeGapWeight`可以在不同行为对之间使用不用的权重。

---