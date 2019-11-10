---
title: Hexo博客文章链接优化
date: 2019-06-11
abbrlink: Hexoblogurloptimize
categories:
- 博客搭建
tags:
- Hexo
- 链接优化
copyright: true
---

## 文章的URL
文章默认的URL配置是包含年月日以及文章标题的，而且每次文章文章有修改就会引起一些链接的变化，繁琐且不易于检索传播。而URL地址对于SEO来说（Search Engine Optimization：搜索引擎优化）是相当重要的，如何缩短并固定每篇文章的连接，同时又可以在链接后面加上`html`使其显得更为正式。这就是本篇文章需要讲解的。

效果可参考[我的博客](www.xiemingzhao.com)，部署环境是`Hexo+Next`。

## 插件安装与配置
基于`Hexo`搭建的博客，可以通过插件`hexo-abbrlink`来实现自定义文章的连接。首先我们使用如下代码进行优化：

```
npm install hexo-abbrlink --save
```

接着打开站点配置文件`_config.yml`，按照如下部分进行相关配置：
```
# URL
## If your site is put in a subdirectory, set url as 'http://yoursite.com/child' and root as '/child/'
url: https://www.xiemingzhao.com
root: /
#permalink: :year/:month/:day/:title/
#permalink_defaults:
permalink: posts/:abbrlink.html
abbrlink:
  alg: crc32  # 算法：crc16(default) and crc32
  rep: hex    # 进制：dec(default) and hex
```

如上所示，是我本人的配置，其中`url`配置的是我博客所关联的域名，具体域名的捆绑可参考我的另一篇博客，[Hexo博客绑定域名](https://www.xiemingzhao.com/posts/Hexoblogdomain.html)。另，`permalink`的配置，我多加了一个固定链接`posts`，纯属个人喜好，你也可以去掉直接配置成`:abbrlink.html`。

## 文章配置与效果
我们完成了如上的配置后，如果不对博客文章做任何处理的话，在部署的时候，将会根据算法随机生成每篇博客的数字链接。当然，如果你觉得使用随机的数字连接不具有识别性，想要自定义每篇博客的链接的话也是可以的，只需要在你的博客`.md`文章的头部配置如下字段即可：

```
abbrlink: your_blog_url
```
例如，我的这篇博客就是配置成`abbrlink: Hexoblogurloptimize`，最终我的博客的连接就是[https://www.xiemingzhao.com/posts/Hexoblogurloptimize.html](https://www.xiemingzhao.com/posts/Hexoblogurloptimize.html)。

通过这一顿操作，你就可以随心所欲控制你的博客的链接了。每次修改博客文章的时候，只要不修改`abbrlink`配置项，那么这篇博客的链接就永远不会发生变化。这样不仅有利于博客链接的记忆与传播，更有利于整个博客的SEO优化，提升检索度和排名。

## 一些官方配置信息
官方文章中你还可以使用如下变量的配置，当然除了这些还可以使用`Front-matter`中的所有属性。

| 变量 | 描述 |
| :--- | :--- |
| `:year` | 文章的发表年份（4 位数） |
| `:month` | 文章的发表月份（2 位数） |
| `:i_month` | 文章的发表月份（去掉开头的零） |
| `:day` | 文章的发表日期 (2 位数) |
| `:i_day` | 文章的发表日期（去掉开头的零） |
| `:title` | 文件名称 |
| `:id` | 文章 ID |
| `:category` | 分类。如果文章没有分类，则是 `default_category` 配置信息。 |

假设你在永久链接中使用一些变量，利于`lang`，你可以在`permalink_defaults`中进行个变量的默认值配置，例如：

```
permalink_defaults:
  lang: en
```

如此，你不指定该变量的时候，就会使用默认值，增加了灵活性。

**参考文章**
[Hexo | 博客文章链接优化](https://zuiyu1818.cn/posts/NexT_seourl.html)
[永久链接（Permalinks）](https://hexo.io/zh-cn/docs/permalinks.html)