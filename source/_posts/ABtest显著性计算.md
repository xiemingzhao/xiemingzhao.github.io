---
title: ABTest显著性计算
categories:
- ABTest
tags:
- ABTest

mathjax: true
copyright: true
abbrlink: ABTestsignificancecomputing
date: 2019-07-16
---

## 显著性计算--uv based

**这里以实验目标为提升CR（Conversion Rate）为例说明**

### 名词解释
**显著性：** 显著：新版和老版的CR有明显差异，不显著: 新版和老版没有明显差异。
**上升幅度：**(新版CR-老版CR)/老版CR
**功效:** 一般功效（即power值）达到0.8, 我们认为样本量即实验UV充足，可下结论。

<!--more-->

>假设观察实验进行3天后，power=0.5<0.8，并且结果不显著，这时需要累计更多样本。 如果当power已达到0.8时，仍未显著，一般我们认为新版和老版的CR的确无明显差异。

**AA校验：**验证主测频道分流是否随机；若两个Control版本之间的指标没有显著差异，则表明分流随机；反之，则需排查Control版本中是否存在异常数据；

>AA异常也可能由于两个control版本，其中之一包含一些异常用户（订单数极高），而另外一个版本没有异常用户。

#### 如何结合显著性、power和样本量对实验结果下结论

power和样本量功能类似，达到样本量基本等同于power达到80%。power与样本量计算相比，power可以更多的利用实验本身的信息，而样本量主要使用频道的数据，计算时与实验设置分流等无关，仅实验剩余天数与实验相关。

所以这里我们结合power和显著性对实验的结果进行判断。这里以转化率CR为例。

>1. 如果power达到80%时，CR仍不显著， 说明此时实验新版与老版无显著差异，停止实验。
2. 如果power未到达80%，CR不显著，说明此时样本量不充足，需继续实验，累计更多的用户。

**以上均基于AA检验正常为前提。**

如果AA异常，需查询原因，如果是AA中某一版本中有少数用户订单数极高，导致AA异常，剔除这种异常用户后重新计算AA  Test的结果， 如果不再显著，AA正常。

严谨一点，再检查AB 的检验中(一般B<新版> vs C+D<C、D都为老版>)是否存在同样问题，即某一版本出现一些异常用户(订单数极高的用户), 如果存在，剔除后重新计算显著性。

### 算法说明

#### **显著性计算**

![ABtestsample](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/Abtest/Abtest-2.png)

我们将指标提升的百分比定义为lift， **%Lift=（Treatment/Control-1）*100%**

如上图的示例结果图所示，CR 的lift估计值为：-0.79%， 区间（-1.55%,-0%）.  CR lift 的真实值以95%的可能性落在区间（-1.55%,-0%）内。由于区间未包括0，所以CR显著, 并且从数值上看是显著下降。说明新版的CR明显低于老版的CR.

具体计算方案，以国内酒店频道的CR例,假设:
>- 老版本每个用户的订单数X为：$x_1,x_2,...,x_{n_1}$，其中$n_1$为老版本的用户数，且有：$E[X] = u_1, Var(X) = \sigma_1^2$
- 新版本每个用户的订单数Y为：$y_1,y_2,...,y_{n_2}$，其中$n_2$为新版本的用户数，且有：$E[Y] = u_2, Var(Y) = \sigma_2^2$
- 尽管 X 和 Y 的分布不满足正态的假设，由大数定律得到，老版人均订单数$(CR_1)$和新版的人均订单数$(CR_2)$分别满足$CR_1 ~ N(u_1，\sigma_1^2 / n_1)$ 和 $CR_2 ~ N(u_2，\sigma_2^2 / n_2)$ 的正态分布。人均订单数即CR。

那么 lift 值的计算方案就如下：
**Step1： 估计$u_1, u_2, \sigma_1^2, \sigma_2^2$**
根据上述四个公式即可得到这四个统计量的估计值。

**Step2：抽样产生 lift 的n（一般取10000）个随机数，$lift^i, i = 1, ..., n$**
由于$CR_1 ~ N(u_1，\sigma_1^2 / n_1), CR_2 ~ N(u_2，\sigma_2^2 / n_2)$，那么结合 Step1 中的参数估计，就可以，

>产生满足$N(\hat u_1, \hat \sigma_1^2 / n_1)$分布的n个随机数，$CR_1^i, i = 1,2,...,n$
产生满足$N(\hat u_2, \hat \sigma_2^2 / n_2)$分布的n个随机数，$CR_2^i, i = 1,2,...,n$

然后我们就可以计算：$lift^i = (CR_2^i - CR_1^i) / CR_1^i, i = 1,2,...,n$

**Step3：计算 lift 的均值和区间（置信度90%）**
lift 均值： $\sum_{i=1}^n lift^i / n$；
区间上界： $lift^i$ 的95%分位数；
区间下界： $lift^i$ 的5%分位数。

#### **功效（power值）计算**
$$Power = \Phi (-Z_{1 - \alpha / 2} + \frac {\Delta}{\sqrt {\sigma_1^2 / n_1 + \sigma_2^2 / n_2} })$$

其中：
>$\alpha$ 是 Type I Error， 一般为0.05 或者0.1
$\sigma_1^2$是老版订单数（或其他指标）的方差，$n_1$是老版的uv数
$\sigma_2^2$是新版订单数（或其他指标）的方差，$n_2$是新版的uv数
$\Delta = lift * u$中的 lift 是实际实验新版相对老板提升的百分比，一般取值为0.02或者0.04，这里设此目标值是为了固定，使用实际的会出现波动太乱的情况
u 是老版的 CR （或者其他检验指标）。

示例：一下以某一次酒店排序实验为例，其 type I error = 0.05, lift = 0.02, 计算 CR 对应的 power。
$n_1$ = 老版用户数 = 22917； $n_2$ = 新版用户数 = 34389
$\hat u$ = 老版 CR 估计值 = 0.37474
$\hat \sigma_1^2$ = 老版订单数方差估计值 = 0.7188733
$\hat \sigma_2^2$ = 新版订单数方差估计值 = 0.721059

$$Power = \Phi (-Z_{1 - \alpha / 2} + \frac {\Delta}{\sqrt {\sigma_1^2 / n_1 + \sigma_2^2 / n_2} }) \\
 =  \Phi (-1.959964 + \frac {0.02 \times 0.37474}{\sqrt {0.7188733/22917 + 0.721059/34389} }) \\
 = \Phi (-0.923327) = 17.79 \%$$

附属代码：
**AbSampleSize.hql**
```sql
set hive.auto.convert.join=true;
set hive.auto.convert.join.noconditionaltask.size =2048;
set hive.exec.parallel=true;

INSERT OVERWRITE TABLE report_abtestdb.AbSampleSize PARTITION(d='${operate_date}',clienttype='${client_type}')
SELECT  exp.experiment,
        exp.channelid,
        cumstart,
        cumend,
        abversion,

        channel.SmallestSize*(splitPct/100) as SmallestSize,
        exp.uv as cumulativeUv,
        datediff(if(datediff('${operate_date}',exp.cumend)<0,'${operate_date}',exp.cumend),exp.cumstart)+1 as days,
        --(channel.SmallestSize/exp.uv*(splitPct/100)-1)*(days)
        (channel.SmallestSize/exp.uv*(splitPct/100)-1)*(datediff(if(datediff('${operate_date}',exp.cumend)<0,'${operate_date}',exp.cumend),exp.cumstart)+1) as remainingDays

FROM report_abtestdb.AbChannelDailyAbsolute channel
JOIN `report_abtestdb`.`AbUserCumulativeAbsolute` exp
ON channel.channelid=exp.channelid
AND channel.d='${operate_date}'
AND channel.clienttype='${client_type}'
AND exp.d='${operate_date}'
AND exp.clienttype='${client_type}'

JOIN (
    SELECT DISTINCT experiment,
                    version,
                    splitPct
    FROM dim_abtestdb.DimAbtestConfig
    WHERE d='${operate_date}'
    AND defaultversion=FALSE 
    AND lower(versionproperty)='treatment'
    AND splitPct>0
) config
ON exp.experiment=config.experiment
AND exp.abversion=config.version;
```

**AbSignificance.hql**
```sql
set hive.auto.convert.join=true;
set hive.auto.convert.join.noconditionaltask.size =2048;
set hive.exec.parallel=true;

add jar abtest_udf-1.0.jar;
add jar commons-math3-3.5.jar;

CREATE TEMPORARY FUNCTION lift_quantile AS 'com.ctrip.basebiz.abtest3.hive.function.UDFLiftQuantile';
CREATE TEMPORARY FUNCTION pnorm AS 'com.ctrip.basebiz.abtest3.hive.function.statistics.UDFCumulativeProbabilityNormalDistribution';
CREATE TEMPORARY FUNCTION makeJson AS 'com.ctrip.basebiz.abtest3.hive.function.UDFMakeJSONObj';

FROM 
-- For Confidence Interval
(
    SELECT  treatment.experiment,
            treatment.channelid,
            treatment.abversion,
            treatment.cumstart,
            treatment.cumend,

            (treatment.mean_pv-control.mean_pv)/control.mean_pv as lift_pv,
            lift_quantile(treatment.mean_pv,treatment.std_pv,control.mean_pv,control.std_pv,0.9,100000) as lift_quantile_pv,
            pnorm(-1.644854+0.02*control.mean_pv/sqrt(power(treatment.stddev_pv,2)/treatment.uv+power(control.stddev_pv,2)/control.uv)) as power_pv,

            (treatment.mean_orders-control.mean_orders)/control.mean_orders as lift_orders,
            lift_quantile(treatment.mean_orders,treatment.std_orders,control.mean_orders,control.std_orders,0.9,100000) as lift_quantile_orders,
            pnorm(-1.644854+0.02*control.mean_orders/sqrt(power(treatment.stddev_orders,2)/treatment.uv+power(control.stddev_orders,2)/control.uv)) as power_orders,

            (treatment.mean_quantity-control.mean_quantity)/control.mean_quantity as lift_quantity,
            lift_quantile(treatment.mean_orders,treatment.std_orders,control.mean_orders,control.std_orders,0.9,100000) as lift_quantile_quantity,
            pnorm(-1.644854+0.02*control.mean_quantity/sqrt(power(treatment.stddev_quantity,2)/treatment.uv+power(control.stddev_quantity,2)/control.uv)) as power_quantity,

            (treatment.dynamicMap['mean_amount']-control.dynamicMap['mean_amount'])/control.dynamicMap['mean_amount'] as lift_amount,
            lift_quantile(treatment.dynamicMap['mean_amount'],treatment.dynamicMap['std_amount'],control.dynamicMap['mean_amount'],control.dynamicMap['std_amount'],0.9,100000) as lift_quantile_amount,
            pnorm(-1.644854+0.02*control.dynamicMap['mean_amount']/sqrt(power(treatment.dynamicMap['stddev_amount'],2)/treatment.uv+power(control.dynamicMap['stddev_amount'],2)/control.uv)) as power_amount,

            (treatment.dynamicMap['mean_cost']-control.dynamicMap['mean_cost'])/control.dynamicMap['mean_cost'] as lift_cost,
            lift_quantile(treatment.dynamicMap['mean_cost'],treatment.dynamicMap['std_cost'],control.dynamicMap['mean_cost'],control.dynamicMap['std_cost'],0.9,100000) as lift_quantile_cost,
            pnorm(-1.644854+0.02*control.dynamicMap['mean_cost']/sqrt(power(treatment.dynamicMap['stddev_cost'],2)/treatment.uv+power(control.dynamicMap['stddev_cost'],2)/control.uv)) as power_cost,

            (treatment.dynamicMap['mean_gross_profit']-control.dynamicMap['mean_gross_profit'])/control.dynamicMap['mean_gross_profit'] as lift_gross_profit,
            lift_quantile(treatment.dynamicMap['mean_gross_profit'],treatment.dynamicMap['std_gross_profit'],control.dynamicMap['mean_gross_profit'],control.dynamicMap['std_gross_profit'],0.9,100000) as lift_quantile_gross_profit,
            pnorm(-1.644854+0.02*control.dynamicMap['mean_gross_profit']/sqrt(power(treatment.dynamicMap['stddev_gross_profit'],2)/treatment.uv+power(control.dynamicMap['stddev_gross_profit'],2)/control.uv)) as power_gross_profit

    FROM `report_abtestdb`.`AbUserCumulativeAbsolute` treatment
    JOIN `report_abtestdb`.`AbUserCumulativeAbsolute` control
     ON treatment.experiment=control.experiment
    AND treatment.channelid=control.channelid
    AND lower(treatment.versionproperty)='treatment'
    AND control.abversion='control'
    AND treatment.DefaultVersion=FALSE 
    AND control.DefaultVersion=FALSE 
    AND treatment.clienttype='${client_type}'
    AND control.clienttype='${client_type}'
    WHERE treatment.d='${operate_date}'
    AND control.d='${operate_date}'
) ciResult
LEFT OUTER JOIN 
-- For AA Test
(
    SELECT  experiment,
            channelid,
            (mean_pv1-mean_pv2)/mean_pv2 as lift_pv,
            lift_quantile(mean_pv1,std_pv1,mean_pv2,std_pv2,0.95,100000) as lift_quantile_pv,

            (mean_orders1-mean_orders2)/mean_orders2 as lift_orders,
            lift_quantile(mean_orders1,std_orders1,mean_orders2,std_orders2,0.95,100000) as lift_quantile_orders,

            (mean_quantity1-mean_quantity2)/mean_quantity2 as lift_quantity,
            lift_quantile(mean_quantity1,std_quantity1,mean_quantity2,std_quantity2,0.95,100000) as lift_quantile_quantity,

            (mean_amount1-mean_amount2)/mean_amount2 as lift_amount,
            lift_quantile(mean_amount1,std_amount1,mean_amount2,std_amount2,0.95,100000) as lift_quantile_amount,

            (mean_cost1-mean_cost2)/mean_cost2 as lift_cost,
            lift_quantile(mean_cost1,std_cost1,mean_cost2,std_cost2,0.95,100000) as lift_quantile_cost,

            (mean_gross_profit1-mean_gross_profit2)/mean_gross_profit2 as lift_gross_profit,
            lift_quantile(mean_gross_profit1,std_gross_profit1,mean_gross_profit2,std_gross_profit2,0.95,100000) as lift_quantile_gross_profit
    FROM (
        SELECT  experiment,
                channelid,
                abversion as i_version,
                FIRST_VALUE(abversion)      over(PARTITION BY experiment,channelid) as abversion1,
                LAST_VALUE(abversion)       over(PARTITION BY experiment,channelid) as abversion2,

                FIRST_VALUE(mean_pv)        over(PARTITION BY experiment,channelid) as mean_pv1,
                LAST_VALUE(mean_pv)         over(PARTITION BY experiment,channelid) as mean_pv2,
                FIRST_VALUE(std_pv)         over(PARTITION BY experiment,channelid) as std_pv1,
                LAST_VALUE(std_pv)          over(PARTITION BY experiment,channelid) as std_pv2, 

                FIRST_VALUE(mean_orders)    over(PARTITION BY experiment,channelid) as mean_orders1,
                LAST_VALUE(mean_orders)     over(PARTITION BY experiment,channelid) as mean_orders2, 
                FIRST_VALUE(std_orders)     over(PARTITION BY experiment,channelid) as std_orders1,
                LAST_VALUE(std_orders)      over(PARTITION BY experiment,channelid) as std_orders2, 

                FIRST_VALUE(mean_quantity)  over(PARTITION BY experiment,channelid) as mean_quantity1,
                LAST_VALUE(mean_quantity)   over(PARTITION BY experiment,channelid) as mean_quantity2, 
                FIRST_VALUE(std_quantity)   over(PARTITION BY experiment,channelid) as std_quantity1,
                LAST_VALUE(std_quantity)    over(PARTITION BY experiment,channelid) as std_quantity2,

                FIRST_VALUE(dynamicMap['mean_amount'])  over(PARTITION BY experiment,channelid) as mean_amount1,
                LAST_VALUE(dynamicMap['mean_amount'])   over(PARTITION BY experiment,channelid) as mean_amount2, 
                FIRST_VALUE(dynamicMap['std_amount'])   over(PARTITION BY experiment,channelid) as std_amount1,
                LAST_VALUE(dynamicMap['std_amount'])    over(PARTITION BY experiment,channelid) as std_amount2,

                FIRST_VALUE(dynamicMap['mean_cost'])  over(PARTITION BY experiment,channelid) as mean_cost1,
                LAST_VALUE(dynamicMap['mean_cost'])   over(PARTITION BY experiment,channelid) as mean_cost2, 
                FIRST_VALUE(dynamicMap['std_cost'])   over(PARTITION BY experiment,channelid) as std_cost1,
                LAST_VALUE(dynamicMap['std_cost'])    over(PARTITION BY experiment,channelid) as std_cost2,

                FIRST_VALUE(dynamicMap['mean_gross_profit'])  over(PARTITION BY experiment,channelid) as mean_gross_profit1,
                LAST_VALUE(dynamicMap['mean_gross_profit'])   over(PARTITION BY experiment,channelid) as mean_gross_profit2, 
                FIRST_VALUE(dynamicMap['std_gross_profit'])   over(PARTITION BY experiment,channelid) as std_gross_profit1,
                LAST_VALUE(dynamicMap['std_gross_profit'])    over(PARTITION BY experiment,channelid) as std_gross_profit2
        FROM `report_abtestdb`.`AbUserCumulativeAbsolute`
        WHERE d='${operate_date}'
        AND clienttype='${client_type}'
        AND lower(versionproperty)='control'
        AND abversion<>'control'
        AND DefaultVersion=FALSE
    ) control
    WHERE i_version=abversion1
    AND abversion1<>abversion2
) aaResult
ON  ciResult.experiment=aaResult.experiment
AND ciResult.channelid=aaResult.channelid

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_pv')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_pv.lci>0 or ciResult.lift_quantile_pv.uci<0) as isSignificant,
        ciResult.lift_pv as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_pv.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_pv.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_pv.lci<=0 AND aaResult.lift_quantile_pv.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_pv.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_pv.lci AS STRING))) as aaExt,
        ciResult.power_pv as `power`,
        makeJson(map()) as powerExt

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_orders')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_orders.lci>0 or ciResult.lift_quantile_orders.uci<0) as isSignificant,
        ciResult.lift_orders as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_orders.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_orders.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_orders.lci<=0 AND aaResult.lift_quantile_orders.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_orders.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_orders.lci AS STRING))) as aaExt,
        ciResult.power_orders as `power`,
        makeJson(map()) as powerExt

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_quantity')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_quantity.lci>0 or ciResult.lift_quantile_quantity.uci<0) as isSignificant,
        ciResult.lift_quantity as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_quantity.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_quantity.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_quantity.lci<=0 AND aaResult.lift_quantile_quantity.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_quantity.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_quantity.lci AS STRING))) as aaExt,
        ciResult.power_quantity as `power`,
        makeJson(map()) as powerExt

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_amount')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_amount.lci>0 or ciResult.lift_quantile_amount.uci<0) as isSignificant,
        ciResult.lift_amount as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_amount.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_amount.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_amount.lci<=0 AND aaResult.lift_quantile_amount.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_amount.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_amount.lci AS STRING))) as aaExt,
        ciResult.power_amount as `power`,
        makeJson(map()) as powerExt

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_cost')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_cost.lci>0 or ciResult.lift_quantile_cost.uci<0) as isSignificant,
        ciResult.lift_cost as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_cost.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_cost.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_cost.lci<=0 AND aaResult.lift_quantile_cost.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_cost.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_cost.lci AS STRING))) as aaExt,
        ciResult.power_cost as `power`,
        makeJson(map()) as powerExt

INSERT OVERWRITE TABLE report_abtestdb.AbSignificance PARTITION(d='${operate_date}',clienttype='${client_type}',index='cr_gross_profit')
SELECT  ciResult.experiment,
        ciResult.channelid,
        ciResult.cumstart,
        ciResult.cumend,
        ciResult.abversion,
        (ciResult.lift_quantile_gross_profit.lci>0 or ciResult.lift_quantile_gross_profit.uci<0) as isSignificant,
        ciResult.lift_gross_profit as lift,
        makeJson(map(
            'uci',CAST(ciResult.lift_quantile_gross_profit.uci AS STRING),
            'lci',CAST(ciResult.lift_quantile_gross_profit.lci AS STRING))
        ) as ciExt,
        (aaResult.lift_quantile_gross_profit.lci<=0 AND aaResult.lift_quantile_gross_profit.uci>=0) as isAANormal,
        makeJson(map(
            'uci',CAST(aaResult.lift_quantile_gross_profit.uci AS STRING),
            'lci',CAST(aaResult.lift_quantile_gross_profit.lci AS STRING))) as aaExt,
        ciResult.power_gross_profit as `power`,
        makeJson(map()) as powerExt
;
```

**UDFCumulativeProbabilityNormalDistribution.java**
```java
package com.ctrip.basebiz.abtest3.hive.function.statistics;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(name = "pnorm",
        value = "_FUNC_(quantile,mean,sd) - Density, distribution function, quantile " +
                "function and random generation for the normal distribution with mean" +
                " equal to mean and standard deviation equal to sd")
public class UDFCumulativeProbabilityNormalDistribution extends UDF {
    public double evaluate(Double quantile) {
        return evaluate(quantile, 0.0D, 1.0D);
    }

    public double evaluate(Double quantile, Double mean, Double sd) {
        if (quantile == null)
            return 0.0;
        NormalDistribution normalDistribution = new NormalDistribution(mean, sd);
        return normalDistribution.cumulativeProbability(quantile);
    }


    public static void  main(String[] args){
        UDFCumulativeProbabilityNormalDistribution d=new UDFCumulativeProbabilityNormalDistribution();
        System.out.println(d.evaluate(0.9));
    }
}
```

**UDFLiftQuantile.java**
```
package com.ctrip.basebiz.abtest3.hive.function;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Collections;
import java.util.PriorityQueue;
import java.util.Random;

@Description(name = "lift_quantile",
        value = "_FUNC_(meanTreatment,stdTreatment,meanControl,stdControl,confidenceLevel,samplingNum).",
        extended = "Construct sets of random number obey Gaussian distribution whose mean and standard deviation " +
                "is the same as treatment version and control version. " +
                "Return lift's upper and lower quantile whose Confidence Level is confidenceLevel.\n" +
                " Lift = (Treatment - Control) / Control ")
public class UDFLiftQuantile extends UDF {
    public static class UDFLiftQuantileResult {
        public DoubleWritable mean;
        public DoubleWritable lci;
        public DoubleWritable uci;
    }

    public UDFLiftQuantileResult evaluate(Double mTreatment, Double stdTreatment,
                                          Double mControl, Double stdControl,
                                          Double confidenceLevel, Integer samplingNum) {
        boolean isArgInvalid = (mTreatment == .0 && stdTreatment == .0) || (mControl == .0 && stdControl == .0);
        if (isArgInvalid) {
            UDFLiftQuantileResult result = new UDFLiftQuantileResult();
            result.mean = new DoubleWritable(0);
            result.lci = new DoubleWritable(0);
            result.uci = new DoubleWritable(0);
            return result;
        }
        Random randomTreatment = new Random();
        Random randomControl = new Random();
        int queueMaxSize = (int) (Math.floor((1.0 - confidenceLevel) / 2 * samplingNum) + 1);
        PriorityQueue<Double> lciQueue = new PriorityQueue<Double>(queueMaxSize, Collections.reverseOrder());
        PriorityQueue<Double> uciQueue = new PriorityQueue<Double>(queueMaxSize);
        BigDecimal sum = BigDecimal.ZERO;
        for (int i = 0; i < samplingNum; i++) {
            double vTreatment = mTreatment + randomTreatment.nextGaussian() * stdTreatment;
            double vControl = mControl + randomControl.nextGaussian() * stdControl;
            if (vControl == 0.0) {
                i--;
                continue;
            }
            double lift = (vTreatment - vControl) / vControl;
            sum = sum.add(BigDecimal.valueOf(lift));
            if (lciQueue.size() < queueMaxSize || lciQueue.peek() >= lift) {
                lciQueue.add(lift);
                if (lciQueue.size() > queueMaxSize) {
                    lciQueue.poll();
                }
            }
            if (uciQueue.size() < queueMaxSize || uciQueue.peek() <= lift) {
                uciQueue.add(lift);
                if (uciQueue.size() > queueMaxSize) {
                    uciQueue.poll();
                }
            }
        }
        UDFLiftQuantileResult result = new UDFLiftQuantileResult();
        result.lci = new DoubleWritable(lciQueue.poll());
        result.uci = new DoubleWritable(uciQueue.poll());
        result.mean = new DoubleWritable(sum.divide(BigDecimal.valueOf(samplingNum), RoundingMode.HALF_EVEN).doubleValue());
        return result;
    }


    public static void main(String[] args) {
        for (int i = 0; i < 10; i++) {
            System.out.println("Nyan~");
            run();
        }
    }

    private static void run() {
        UDFLiftQuantile o = new UDFLiftQuantile();
        UDFLiftQuantileResult result = o.evaluate(0.666141754, 0.257336891, 0.373081111, 0.079436106, 0.8, 100000);
        //UDFLiftQuantileResult result = o.evaluate(.0, .0, .0, .0, 0.8, 1000);
        //UDFLiftQuantileResult result = o.evaluate(.0, .0, 1.0, 2.0, 0.8, 1000);
        //UDFLiftQuantileResult result = o.evaluate(2.0, 54.0, .0, .0, 0.8, 1000);
        System.out.println(result.mean);
        System.out.println(result.lci);
        System.out.println(result.uci);
    }
}
```

## 显著性计算--date based

### 方法简述
当总体呈现正态分布且总体标准差未知，而且容量小于30，那么这时一切可能的样本平均数与总体平均数的离差统计量呈T分布。

该方法采用统计中的two sample t test， 检验两组数据的均值是否相等。例如100个男生身高数据和100个女生身高数据，通过该方法可以检验男生的平均身高是否显著不等于女生的平均身高。

在报表中我们输入的两组数据，一组是一个版本每日的指标数据，另外一组是选择的另外一个版本对应的每日的指标数据。指标可以是任意数值型指标，比如UV数，点击率，订单数等等。

该方法我们只做两两间的比较。

### 实验举例
我们以一个首页改版为例，如下图所示：

![AB home page](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/Abtest/Abtest-1.png)

首页改版，通过上面图片发现海外酒店位置发生变化，所以我们想知道位置的改变是否会影响海外酒店的点击数量，分流比：新版：老版=50%：50%

我们拿到了每天新老版本的点击UV数，通过统计检验，判断是否新版的点击UV数明显低于老版。
数据如下：

![home AB outcome](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/algorithm/Abtest/Abtest-5.png)

### 检验方法
假设x：新版每日UV数，y：老板每日UV数。计算如下统计量：
$$t = \frac {\bar x - \bar y}{s \sqrt{1 / n_1 + 1 / n_2} }$$

这里，
>$\bar x$ 是新版均值，$\bar x = \sum_{i=1}^{n_1} x_i / n_1, n_1 是天数$
$\bar y$ 是老板均值，$\bar y = \sum_{i=1}^{n_1} y_i / n_2, n_2 是天数$
$s = \sqrt{[(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2] / (n_1 + n_2 -2)}$，其中
$s_1^2 = \sum_{i=1}^{n_1} (x_i - \bar x)^2 / (n_1 - 1), s_2^2 = \sum_{i=1}^{n_2} (y_i - \bar y) / (n_2 - 1)$

若 $| t | > t_{n_1 + n_2 - 2, 1 - \alpha / 2}$，则显著，否则不显著。$t_{n_1 + n_2 - 2, 1 - \alpha / 2}$数值可通过查表或者计算器获取。

### 示例剖析
基于上述方案，我们对前面的例子进行计算有：
$\bar x = 7555.111, \bar y = 14935$
共有9天数据，所以$n_1 = n_2 = 9$

$s_1^2 = 1556811, s_2^2 = 335096.8, s = 972.6015$
t = (7555.111-14935) / 3890.406/$\sqrt{2/9}$ = -16.09612

查表或者计算机可得：$t_{n_1 + n_2 - 2, 1 - \alpha / 2} = t_{16,1 - \alpha / 2} = 1.745884（自由度=9+9-2=16）$
由于$| t | > t_{16,1 - \alpha / 2}$，所以新版还外加酒店宫格点击用户数相对老版是显著下降的。

最小样本量：
在方法简述中提到：当总体呈现正态分布且总体标准差未知，而且容量小于30，那么这时一切可能的样本平均数与总体平均数的离差统计量呈T分布。

在ABtest中实验天数小于30天即可用T检验来进行判定。那么是不是实验天数越小越好呢？答案显然是否定的，实验天数越多得到的结论可靠性越好，

但是业务人员希望实验天数越少越好，两者之间形成了悖论。在此，一般建议实验最少进行两周（14天）：一周数据（7天）太少，且旅游数据大部分都是以一周为一个周期上下浮动，选择两周可以有效地平滑掉周期对结果的影响。

### 实验最小 uv 量
假设实验组分流比例 (B) = 对照组分流比例 (C+D), 指标(CR等)满足正态分布(Central Limit Theorem)且方差相等。

选择参与实验的主指标数量为m (选项有CR, Quantity, GP-C)。对于每个选中的主指标, 计算该指标需要的最小样本量$S_i$:

$$n = \frac {((k+1) + (k+1)/k) \sigma^2 (z_{a - \alpha / 2m} + z_{1 - \beta})^2}{\Delta^2}
= treatment_uv + control_uv$$

>- $\Delta = lift * u_x$，大流量 lift 可取值0.02，小流量可取0.03
    （$u_x$可取该指标在试验频道前2周的均值；$lift = (u_y - u_x)/u_x$，其中$(u_y - u_x)$是实验组和对照组的均值差）
- Type I Error 一般取$\alpha = 10 \%$；Type II Error 一般取$\beta = 0.2(Power = 10 \%)$
- $z_x$是正态分布累计概率为 x 时对应的分位数
- $\sigma^2$是该指标子啊试验频道前2周的方差。k = 实验组UV/对照组UV

**最后选取实验的所需最小样本量的最大值$max{S_i: i = 1, ..., m}$**


#### 知识小科普：

T检验和成对T检验的区别：
通常T检验或成对T检验是用来判断两组数据的平均值是否在统计上有差别,换一个理解,对两组数据而言,每组数据本身内部有一个波动范围(组内变异),而两组数据之间平均值的波动相称为组间变异,如果组间变异相对于组内变异小的话,就可以认为两组数据之间的平均值是没有差异的,这是T检验的做法. 而对于成对T检验,在一组中的数据与另一组的数据有对应关系,也就是两组数据是以成队的形式出现的,这个时候,运用这两个成队数据之间的差值,可以得到一个数据列,如果这个数据列的平均值在统计上是非零的,即可认为两组数据均值是有差异的,在这个地方,没有单独的去考虑两组数据之内的差异,而是通过将两组数据中对应的数据相减,得到一组数据,通过类似偏倚的算法,来看它在统计是是否非零.换一句话说,是当组内差异比较大(或者说是噪音较大),但是可以通过其它一个因子作区隔时,可以用成对T检验。