---
title: Github+Hexo+Next博客的编辑方法小记
categories:
- 博客搭建
tags:
- Hexo
- markdown
- mathjax
copyright: true
abbrlink: GithubHexoNexttips
date: 2019-05-19
---

想要小白详细版本的Github+Hexo+Next搭建博客教程，可访问我的另一篇博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/GithubHexoNextblog)。

## 博客撰写

### 1. Markdown
Hexo是基于标准的`Markdown`格式进行解析博客文章的，`markdown`应该不用多说了，如果你除了office全家桶外不知道全加的编辑器也没关系，学习起来很简单，网上教程比比皆是，看看语法就学会了。然后你就会爱上这类文档编辑器，什么word那些妖艳贱货都是不能比的，你会因此爱上写作。其语法简洁易学，可参考[markdown官方中文文档](https://markdown-zh.readthedocs.io/en/latest/)。

当然，它只是一个通用型很高的文档编辑器，而Hexo博客的头部包含一些并不是很通用的格式模块，例如`tile`,`tags`以及`date`等等，这一块在创建博客的文章用已经阐述过，或者去Hexo的官方文档[https://hexo.io/zh-cn/](https://hexo.io/zh-cn/)一探究竟。还有一个需要提的是，markdown本身的语法在标题上面用#来区分，无其他要求，但是hexo解析的时候要求标题类的#与文字之间要有一个空格，反正养成一个良好的习惯，**格式符号与文字之间保留空格。**

<!--more-->

### 2. 编辑器
提到这里，`Markdown`本身只是一种编辑格式，那么真正的编辑器用什么呢，可能你已经有自己喜欢的编辑器了，这里推荐几款个人绝对很好用的编辑器仅供参考：

>[Cmd Markdown](https://www.zybuluo.com/cmd/)，作业部落出品，本人使用最多的，强推
[MarkdownPad](http://markdownpad.com/)，使用免费的版本 MarkdownPad2 足以
[小书匠](http://soft.xiaoshujiang.com/)，也分为分为免费版和收费版，文件管理功能强
[Marxico](http://marxi.co/)，中文名马克飞象，可以直接把文本存到印象笔记

*更多的可以参考[这里](https://blog.csdn.net/qq_36759224/article/details/82229243)*

### 3. 图床
既然要写作，在这个信息爆炸的时代，大家都很赶时间，一篇只有文字的博客难免使人心生倦意，图文并茂的方式往往更好。那么博客附属图片常常食用的就是图床，而一般的网盘是不能够生成外链的，这里推荐几款好用的 **免费** 图床：

>[postimg](https://postimages.org)，本人在用，国外的图床，但是速度也很快。永久存储免注册，图片链接支持https，可以删除上传的图片，提供多种图片链接格式。
[七牛云](https://portal.qiniu.com)，中文名七牛云，注册认证后有10G永久免费空间，每月10G国内和10G国外流量，速度相当快，需要实名认证较为麻烦。
[极简图床](http://jiantuku.com)，不太熟悉，据说是主要提供图片上传和管理界面，需要用户自己设置微博、七牛云或者阿里云OSS信息。

### 4. 公式
作为技术博客，就难免会使用公式，相信经历过word中的公式编辑mathtype的难友们一定难以忘怀其如恶魔般的体验。编辑一个公式，点来点去费时费力，修改和转移都复杂得多，关键是正版的还需要注册收费。而`markdown`大法就完美地解决了这一问题，其内部支持`Latex`公式编辑格式，行内编辑可用`$latex公式$`格式来操作，地理成行的则用双刀符号`$$latex公式$$`格式。举例：

1. 这里展示行内公式，开始\$f(x) = \sum_{i=1}^m \frac{1}{x_i}\$结束。结果就会如下：
 这里展示行内公式，开始$f(x) = \sum_{i=1}^m \frac{1}{x_i}$结束。
2. 这里展示行外独立公式，开始\$\$f(x) = \sum_{i=1}^m \frac{1}{x_i}\$\$结束。结果就会如下：
 这里展示行内公式，开始$$f(x) = \sum_{i=1}^m \frac{1}{x_i}$$结束。
**双$符号就是行外公式，公式本身会独立成一行**

Latex公式编辑语法也不难学，可参考其[Latex官方网站](https://www.latex-project.org/)的语法指南，分分钟让你爱上公式编辑，再也不会害怕编辑公式了。

**参考博文**
[可能是最详细的 Hexo + GitHub Pages 搭建博客的教程](https://blog.csdn.net/qq80583600/article/details/72828063)
[最新主流 Markdown 编辑器推荐](https://blog.csdn.net/qq_36759224/article/details/82229243)
