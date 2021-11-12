---
title: 一致性哈希算法(Consistent Hashing)
categories:
- 学习笔记
- 算法总结
tags:
- Hash
- 算法
mathjax: true
copyright: true
abbrlink: consistentHash
date: 2020-04-08
---

## 1 背景
一致性哈希算法(Consistent Hashing)在分布式系统的应用还是十分广泛的。例如随着业务的扩展，流量的剧增，单体项目逐渐划分为分布式系统。对于经常使用的数据，我们可以使用Redis作为缓存机制，减少数据层的压力。因此，重构后的系统架构如下图所示：

![consistentHash1](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash1.png)

这个时候一般有两种方案：

<!--more-->

1. 每个机器缓存全量数据；
>如此虽然能保证请求打到任何一台机器都可以，但是冗余性巨高；

2. 每个机器只缓存一部分，分布式存储；
>如此需要保证对应的请求打到对应的机器上，否则查询结果为空，轮训查询的话效率极低不靠谱；

## 2 原始Hash方案
这时候，自然而然想到一个方案，那就是使用hash算法 例如，有三台Redis，对于每次的访问都可以通过计算hash来求得hash值。 如公式 h=hash(key)%3，我们把Redis编号设置成0,1,2来保存对应hash计算出来的值，h的值等于Redis对应的编号。

但是，使用上述HASH算法进行缓存时，会出现一些缺陷。例如：
* 缓存的3台缓存服务器由于故障使得机器数量减少；
* 缓存量的增加需要新增使得缓存机器增加。

如此缓存位置必定会发生改变，以前缓存的图片也会失去缓存的作用与意义，由于大量缓存在同一时间失效，造成了缓存的雪崩，此时前端缓存已经无法起到承担部分压力的作用，后端服务器将会承受巨大的压力，整个系统很有可能被压垮，所以，我们应该想办法不让这种情况发生，但是由于上述HASH算法本身的缘故，使用取模法进行缓存时，这种情况是无法避免的，为了解决这些问题，一致性哈希算法诞生了。

## 3 一致性Hash算法
根据上述我们知道一致性Hash算法需要解决：

1. 当缓存服务器数量发生变化时，会引起缓存的雪崩，可能会引起整体系统压力过大而崩溃（大量缓存同一时间失效）。
2. 当缓存服务器数量发生变化时，几乎所有缓存的位置都会发生改变，尽量减少受影响的缓存。


其实，一致性哈希算法也是使用取模的方法，只是，刚才描述的取模法是对服务器的数量进行取模，而一致性哈希算法是对2^32取模。

首先，我们把二的三十二次方想象成一个圆，就像钟表一样，钟表的圆可以理解成由60个点组成的圆，而此处我们把这个圆想象成由2^32个点组成的圆，示意图如下：

![consistentHash2](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash2.png)

我们把这个由2的32次方个点组成的圆环称为hash环。


假设我们有3台缓存服务器，服务器A、服务器B、服务器C，那么，在生产环境中，这三台服务器肯定有自己的IP地址。

**hash（服务器A的IP地址） %  2^32**

通过上述公式算出的结果一定是一个0到2^32-1之间的一个整数，我们就用算出的这个整数，代表服务器A，既然这个整数肯定处于0到2^32-1之间，那么，上图中的hash环上必定有一个点与这个整数对应，而我们刚才已经说明，使用这个整数代表服务器A，那么，服务器A就可以映射到这个环上，用下图示意：

![consistentHash3](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash3.png)

同理，服务器B与服务器C也可以通过相同的方法映射到上图中的hash环中

**hash（服务器B的IP地址） %  2^32**

**hash（服务器C的IP地址） %  2^32**

通过上述方法，可以将服务器B与服务器C映射到上图中的hash环上，示意图如下：

![consistentHash4](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash4.png)

我们通过上述方法，把缓存服务器映射到了hash环上，那么使用同样的方法，我们也可以将需要缓存的对象映射到hash环上。

假设，我们需要使用缓存服务器缓存图片，而且我们仍然使用图片的名称作为找到图片的key，那么我们使用如下公式可以将图片映射到上图中的hash环上。

**hash（图片名称） %  2^32**

映射后的示意图如下，下图中的橘黄色圆形表示图片：

![consistentHash5](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash5.png)

现在服务器与图片都被映射到了hash环上，上图中的图片将会被缓存到服务器A上，为什么呢？因为从图片的位置开始，沿顺时针方向遇到的第一个服务器就是A服务器，所以，上图中的图片将会被缓存到服务器A上，如下图所示：


![consistentHash6](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash6.png)

一致性哈希算法就是通过这种方法，判断一个对象应该被缓存到哪台服务器上的，将缓存服务器与被缓存对象都映射到hash环上以后，从被缓存对象的位置出发，沿顺时针方向遇到的第一个服务器，就是当前对象将要缓存于的服务器，由于被缓存对象与服务器hash后的值是固定的，所以，在服务器不变的情况下，一张图片必定会被缓存到固定的服务器上，那么，当下次想要访问这张图片时，只要再次使用相同的算法进行计算，即可算出这个图片被缓存在哪个服务器上，直接去对应的服务器查找对应的图片即可。

![consistentHash7](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash7.png)

1号、2号图片将会被缓存到服务器A上，3号图片将会被缓存到服务器B上，4号图片将会被缓存到服务器C上。

## 4 一致性哈希算法的优点
一致性哈希算法如何解决之前出现的问题呢？

假设，服务器B出现了故障，我们现在需要将服务器B移除，那么，我们将上图中的服务器B从hash环上移除即可，移除服务器B以后示意图如下：

![consistentHash8](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash8.png)

在服务器B未移除时，图片3应该被缓存到服务器B中，可是当服务器B移除以后，按照之前描述的一致性哈希算法的规则，图片3应该被缓存到服务器C中，因为从图片3的位置出发，沿顺时针方向遇到的第一个缓存服务器节点就是服务器C，也就是说，如果服务器B出现故障被移除时，图片3的缓存位置会发生改变：

![consistentHash9](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash9.png)

但是，图片4仍然会被缓存到服务器C中，图片1与图片2仍然会被缓存到服务器A中，这与服务器B移除之前并没有任何区别，这就是一致性哈希算法的优点，如果使用之前的hash算法，服务器数量发生改变时，所有服务器的所有缓存在同一时间失效了，而使用一致性哈希算法时，服务器的数量如果发生改变，并不是所有缓存都会失效，而是只有部分缓存会失效，前端的缓存仍然能分担整个系统的压力，而不至于所有压力都在同一时间集中到后端服务器上。

## 5 hash环的偏斜
在介绍一致性哈希的概念时，我们理想化的将3台服务器均匀的映射到了hash环上，如下图所示：

![consistentHash10](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash4.png)

在实际的映射中，服务器可能会被映射成如下模样。

![consistentHash11](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash11.png)

如果服务器被映射成上图中的模样，那么被缓存的对象很有可能大部分集中缓存在某一台服务器上，如下图所示。

![consistentHash12](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash12.png)

上图中，1号、2号、3号、4号、6号图片均被缓存在了服务器A上，只有5号图片被缓存在了服务器B上，服务器C上甚至没有缓存任何图片，如果出现上图中的情况，A、B、C三台服务器并没有被合理的平均的充分利用，缓存分布的极度不均匀，而且，如果此时服务器A出现故障，那么失效缓存的数量也将达到最大值，在极端情况下，仍然有可能引起系统的崩溃，上图中的情况则被称之为hash环的偏斜。当然也有办法解决此问题的。

## 6 虚拟节点
话接上文，由于我们只有3台服务器，当我们把服务器映射到hash环上的时候，很有可能出现hash环偏斜的情况，当hash环偏斜以后，缓存往往会极度不均衡的分布在各服务器上，聪明如你一定已经想到了，如果想要均衡的将缓存分布到3台服务器上，最好能让这3台服务器尽量多的、均匀的出现在hash环上，但是，真实的服务器资源只有3台，我们怎样凭空的让它们多起来呢，没错，就是凭空的让服务器节点多起来，既然没有多余的真正的物理服务器节点，我们就只能将现有的物理节点通过虚拟的方法复制出来，这些由实际节点虚拟复制而来的节点被称为”虚拟节点”。加入虚拟节点以后的hash环如下。

![consistentHash13](http://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/papers/consistentHash13.png)

虚拟节点”是”实际节点”（实际的物理服务器）在hash环上的复制品,一个实际节点可以对应多个虚拟节点。

从上图可以看出，A、B、C三台服务器分别虚拟出了一个虚拟节点，当然，如果你需要，也可以虚拟出更多的虚拟节点。引入虚拟节点的概念后，缓存的分布就均衡多了，上图中，1号、3号图片被缓存在服务器A中，5号、4号图片被缓存在服务器B中，6号、2号图片被缓存在服务器C中，如果你还不放心，可以虚拟出更多的虚拟节点，以便减小hash环偏斜所带来的影响，虚拟节点越多，hash环上的节点就越多，缓存被均匀分布的概率就越大。

**参考文章**
[白话解析：一致性哈希算法 consistent hashing](https://www.zsythink.net/archives/1182)
[5分钟理解一致性哈希算法](https://juejin.cn/post/6844903750860013576)