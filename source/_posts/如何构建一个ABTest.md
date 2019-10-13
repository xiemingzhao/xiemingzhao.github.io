---
title: 如何构建一个ABTest
categories:
- ABTest
tags:
- ABTest

mathjax: true
copyright: true
abbrlink: ABTestbuilding
date: 2019-07-12
---

## 如何构建一个ABTest

### 谁会参与
1. 业务部门。 定义一个Ab Testing.
2. Ab Testing管理后台。 录入一个Ab Tesing.
3. 业务开发者。 负责具体的方案开发实现。
4. 用户。我们所关心的用户。
5. 数据收集系统。收集我们关心的数据。
6. 数据分析系统。根据收集到的数据，分析出我们所关心的指标。

<!--more-->

### 构建一个Ab Testing的流程

>**简要步骤**
![ab-testing-lifecycle-summary](https://i.postimg.cc/qMbXmmB8/ab-testing-lifecycle-summary.jpg)

1. 业务部门负责定义一个ab testing
2. 管理后台负责录入管理ab testing
3. 业务开发负责具体的业务实现
4. 日志系统负责收集数据
5. 数据分析系统负责生成指定报表

Ab Testing Framework 在其中的作用：
1. 无数据管理，提供restful api接口
2. 提供client library 供业务开发使用，实现分流。


### 流程细化
![ab-testing-lifecycle](https://i.postimg.cc/0yhGvbwj/ab-testing-lifecycle.png)

### Ab Testing Framework
![where-is-ab-testing-framework](https://i.postimg.cc/kXXvvrCJ/where-is-ab-testing-framework.png)

### AB Test 基础方面

**1. ab配置系统环境**
由于生产环境的防火墙限制，实验不能往生产环境同步。建议实验先配置生产环境，然后往fws环境和uat环境进行同步。

**2. 实验状态**
>a.目前一个实验有配置中、待审核、审核通过（进行中）、实验结束等状态。
b.审核通过之后到了实验开始时间即为进行中。其中，实验开始时间以计算开始时间为准。
c.同步过去的实验状态为配置中。同步实验不能同步实验的分流规则，没有分流规则的实验一定是配置中状态。
d.实验的版本数有改动会清空规则，实验重新进入配置中状态。

**3. 实验类型**
借助于浏览器的是web实验，web实验里面分为online实验和h5实验，区分在于是否采用h5技术；借助于app的是app实验，app实验里面分为app native、app hybrid和app服务端实验。采用手机原生态接口的，或者建立在原生态接口之上的是app native实验；其中，再混入h5或其他技术，是hybrid实验。

|| online实验 | h5实验 | app native实验 | app hybrid实验 | app 服务端实验 |
| :---: | :--- | :--- | :--- | :--- | :--- | 
| 实验位置 | 客户端	| 客户端 | 客户端 |	客户端 | 服务端 |
| 采用技术 | html、css、js等web技术 | 主要为h5技术，web app | native app |	hybrid app |	服务端技术 |

最容易混淆的是h5实验和app hybrid实验，需要注意，这两个的区别在于h5实验直接借助浏览器呈现实验页面，而app hybrid实验借助于携程app，直接或间接的采用系统的原生态接口。也可参考H5 & Native & Hybrid。

**4. ab实验版本**

- 版本号。abtest测试至少需要4个版本号，3个版本号放置老版，1个版本号放置新版。如果有多个新版，可在这基础上新增版本号代表新版。老版为control，新版为treatment。、
- 版本开多少流量。在4个版本ABCD中，B为新版，ACD为老版。其中，建议A老版，默认版，盛放流量余量；B新版，新版流量；CD老版，AA实验流量。流量关系需要满足1.sum(A+B+C+D)=100%；2.C=D；3.C+D=B。
- 不建议实验上线之后还修改版本。如果修改了版本，请重新分配流量。

**5. 分流**

1. 分流比是否正常
可通过dashboard监控来看分流比，其中metricname为abtest.client.reqeust.count,expcode选择实验号，并按expversion分组。在请求数达到一定量的前提，如果存在分流比和配置的分流比出入较大，即分流异常。
2. 分流比异常原因
A. 分流总体较少；B. 实验相互干扰；C.实验存在嵌套（目前不支持）；D.配置或ab代码有误
3. app实验分流请求总数很大
该请求数表示直接调用abtest.client.dll的接口数，不反映app实验的用户分流请求数。基于app渠道的特殊性，只要用户启动app，将在app加载所有app实验的的ab分流结果。

### 异常数据剔除

异常用户判断逻辑：
1.取出所有μ+3Sigma以外的，放入集合A;
2.对集合A中的用户做LOF（Local Outlier Factor），取出离群点放入集合B；
3.取集合B中用户，其数值大于μ+4Sigma的，视为异常用户。

**这里可根据多个维度进行判断，例如pv量，order量以及amount量**

异常数据剔除逻辑：
1.取出异常用户，将该用户在实验有效期内参与的数据均做剔除；