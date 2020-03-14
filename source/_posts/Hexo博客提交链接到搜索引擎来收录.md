---
title: Hexo博客提交链接到搜索引擎来收录
date: 2019-06-15
abbrlink: HexoblogSE
categories:
- 博客搭建
tags:
- Hexo
- 网站收录
copyright: true
---

## 写在前面
博客的搭建和个性化可以参考我的其他文章[Hexo搭建博客汇总](https://www.xiemingzhao.com/tags/Hexo/)。当你博客搭建完毕后，如果不能被人搜索得到，心里难免会有些失落。所以，接下来我们介绍 Google 和百度收录博客网站的方法。整体来说，Google 实在是太效率了，收录操作不仅简单且迅速，基本一个小时内就可以检索了。相比之下，百度搜索则鸡肋的很，不仅操作繁杂，而且及时操作成功了收录成功与否还去取决于网站质量以及其其他原因。

**首先如何检测自己的博客能否被检索呢？**
在百度或者Google的搜索框内输入以下内容：

```
site:www.xiemingzhao.com
```

将`site:`后面的网址改为你自己的博客地址就行了，如果在搜索结果中能够展示自己博客的页面，那么就说已经被收录且可被搜索到。反之，则没有被收录。

<!--more-->

## Google 收录
搜索网站的收录，其实就是将网站里各个网页对应的连接收录。所以，有一个东西就叫做站点地图，顾名思义，就是将自己网站下所有的页面集中到一起。

### 安装站点地图
我们需要安装以下插件来生成站点地图：

```
npm install hexo-generator-sitemap --save
npm install hexo-generator-baidu-sitemap --save  
```

可以看得出来，上面包含两个工具包，因为后面也是进行百度收录，而百度的站点地图格式与Google是有差异的，所以一次性将这两个全都安装了。

然后我们打开站点配置文件，找到或者添加如下的配置：

```
#hexo sitemap
sitemap:
  path: sitemap.xml
baidusitemap:
  path: baidusitemap.xml
```
*实际上，在操作中发现只要保留上面的`sitemap`配置，省略下面的也能生成两个*

到此，后面再部署博客的时候，你会发现`public`目录下面多了 `sitemap.xml` 和 `baidusitemap.xml` 两个文件，同样的在线上也可添加这个页面，例如我的就是[我的站点地图](https://www.xiemingzhao.com/sitemap.xml)。

>注意：
1. 插件生成的 sitemap 的文章链接，都是以站点配置文件中的 url 为基础的，如果将博客绑定了域名，那最好将 url 字段填写为绑定的域名。
2. 不想生成 sitemap 的页面，可在页面最上方以 --- 分割的区域内，即 Front-matter 中，添加代码 sitemap: false。

### 添加 robots.txt
>robots.txt（统一小写）是一种存放于网站根目录下的 ASCII 编码的文本文件，它通常告诉网络搜索引擎的漫游器（又称网络蜘蛛），此网站中的哪些内容是不应被搜索引擎的漫游器获取的，哪些是可以被漫游器获取的。

在 `source` 目录下增加 `robots.txt` 文件， 我的文件具体内容如下可供参考，注意将域名改为自己的网站：

```
User-agent: *
Allow: /
Allow: /archives/
Allow: /tags/
Allow: /categories/
Allow: /about/
Allow: /guestbook/
Allow: /others/


Disallow: /js/
Disallow: /css/
Disallow: /lib/

Sitemap: https://www.xiemingzhao.com/sitemap.xml
Sitemap: https://www.xiemingzhao.com/baidusitemap.xml
```
这样在下次部署博客时，`robots.txt` 就会被上传至网站了。稍后我们在提交 `sitemap` 时，可以顺便测试它是否被搜索引擎正确解析了。

### 提交站点到 Google
我们打开[Google 的站点平台](https://www.google.com/webmasters/tools/home?hl=zh-CN)。你会看到如下页面，紧接着就是注册和登录，你有账号的话直接登录都可以。

![se1](https://i.postimg.cc/xjJ40fbg/se1.jpg)

紧接着，点击左上角的`添加资源`，开始验证自己的博客网站，你会看到如下页面，这里建议选择第二个，直接诸如博客站点的主链接就行了，例如我的就是`https://www.xiemingzhao.com`。点击继续后，需要你做一个很简单的验证方式，那就是将验证`html`文件下周下来之后放到自己博客站点的根目录上，然后可以部署一下。

![se2](https://i.postimg.cc/26J94Jg4/se2.jpg)

之后回到验证的页面，点击验证即可，验证程刚后，就可以对你的博客站点进行站点地图的提交了。我们点击左侧的`站点地图`选项，你会看到如下的页面，在这里输入前面构建好的`sitemap`的地址再提交就可以了。

![se3](https://i.postimg.cc/jjMFjCX3/se3.jpg)

到这里就完成了 Google 的检索收录，是不是超级简单。稍等一段时间，就可以去Google上面进行测试自己的博客站点，我的在一个小时内就已经能够检索到了。

## 百度的收录
百度的收录相比对Google要复杂的多，首先需要注册[百度站长平台](https://ziyuan.baidu.com)。目前，账号需要绑定熊掌号才可以，因为熊掌号是百度的资源实名认证和管理的方法，不过对于我们来说确实麻烦了好多。这一些都很简单，就是费一些时间，当你实名认证并且绑定好账号之后，让我们开始下面的提交收录。

这里附上[百度站长工具平台使用帮助手册](https://ziyuan.baidu.com/college/courseinfo?id=267&page=2)，如果你有兴趣深入研究，可以仔细研究一下这个手册，里面基本上包含了各种坑的解决方法，缺点就是繁琐。

### 验证网站
首先需要的就是验证网站，我们进入[站点管理](https://ziyuan.baidu.com/site/siteadd)，添加自己的博客站点地址，然后一步一步往下点，会有一些让你选择与你博客有关系的问题，客观的选择就可以了。

直到最后一步，网站的验证，如下所示，百度的验证有很多种方案，但我们跟上述Google一样，选择文件验证，方便又高效，下载对应的`html`文件，放入博客的根目录，部署后，访问一下试试，可以的话，回到验证页面，点击验证即可完成，基本上没什么难度。

>百度站长平台提供三种验证方式（百度统计的导入方式已下线）：文件验证、html标签验证、CNAME验证。
1.文件验证：您需要下载验证文件，将文件上传至您的服务器，放置于域名根目录下。
2.html标签验证：将html标签添加至网站首页html代码的<head>标签与</head>标签之间。
3.CNAME验证：您需要登录域名提供商或托管服务提供商的网站，添加新的DNS记录。

### 链接提交
接下来最终要就是这一步，但也是最复杂方法最多的，不着急我们慢慢来。首先，进入站长平台，然后进入`网页抓取`目录下的`链接提交`页面，我们可以看到如下图，数据提交方式下面紧邻的是提交连接数量，目前你可得是没有的。再往下，你会发现有两个模块，分别是`自动提交`和`手动提交`。

其中收到提交点击去可以发现是需要你将你需要被检索的网站一个一个的列出来才行，这种笨方法当然不是我们要选择的。让我们回来看自动提交下面，分为三种方法，其中主动推送(实时)又分为四中示例，整个结构如下：

```
链接提交：
    手动提交
    自动提交：
        主动推送（实时）：
            curl推送
            post推送
            php推送
            ruby推送
        自动推送
        sitemap
```

![se4](https://i.postimg.cc/J0k6KxWS/se4.jpg)

### sitemap 提交
在上面，我们已经构建了`baidusitemap`了，在这里当然要使用了。我们选择自动提交中的`sitemap`，输入自己的`baidusitemap.xml`链接即可，一般都是自己的域名加上这个，例如我的就是[https://www.xiemingzhao.com/baidusitemap.xml](https://www.xiemingzhao.com/baidusitemap.xml)。提交完成后可查看是否成功。

>注意：和谷歌不同，百度翻译速度很慢，而且百度提交了链接也不一定收录，要不断提升文章质量和数量才行。

### 百度相关的搜索配置
由于 GitHub 屏蔽了百度的爬虫，即使提交成功，百度知道这里有可供抓取的链接，也不一定能抓取成功。首先我们先检测一下百度爬虫是否可以抓取网页。在百度站长平台`网页抓取`->`抓取诊断` 中，选择`PC UA`点击抓取，查看抓取状态，如果显示`抓取失败`，则需要进一步的配置。

可以测试一下`移动 UA`，因为一般这个一定是会成功的。

### 主动推送和自动推送
我们讲过，百度`sitemap`的提交不一定能够成功，而且即使成功效率也低。百度本身也不提倡，所以还有另两种方案。

在前面提到的百度站长手册中，有讲解这一切，包括如何选择链接提交方式
>1、主动推送：最为快速的提交方式，推荐您将站点当天新产出链接立即通过此方式推送给百度，以保证新链接可以及时被百度收录。
2、自动推送：最为便捷的提交方式，请将自动推送的 JS 代码部署在站点的每一个页面源代码中，部署代码的页面在每次被浏览时，链接会被自动推送给百度。可以与主动推送配合使用。
3、sitemap：您可以定期将网站链接放到 sitemap 中，然后将 sitemap 提交给百度。百度会周期性的抓取检查您提交的 sitemap，对其中的链接进行处理，但收录速度慢于主动推送。
4、手动提交：一次性提交链接给百度，可以使用此种方式

#### 自动推送
next 主题已经部署了自动推送的代码，我们只需在主题配置文件 中找到 `baidu_push` 字段 , 设置其为 true 即可。

#### 主动推送（实时）
这个方案好处在于成功率大，且具有实时性。可以参考这篇文章[ Hexo 插件之百度主动提交链接](https://hui-wang.info/2016/10/23/Hexo%E6%8F%92%E4%BB%B6%E4%B9%8B%E7%99%BE%E5%BA%A6%E4%B8%BB%E5%8A%A8%E6%8F%90%E4%BA%A4%E9%93%BE%E6%8E%A5/)。

![se5](https://i.postimg.cc/1t4YVP7m/se5.jpg)

首先我们在如上图中找到自己的秘钥，保存留用。紧接着，我们需要安装以下插件：
```
npm install hexo-baidu-url-submit --save
```

然后，同样在根目录下，把以下内容配置到_config.yml文件中:

```
baidu_url_submit:
  count: 1 ## 提交最新的一个链接
  host: www.hui-wang.info ## 在百度站长平台中注册的域名
  token: your_token ## 请注意这是您的秘钥， 所以请不要把博客源代码发布在公众仓库里!
  path: baidu_urls.txt ## 文本文档的地址， 新链接会保存在此文本文档里
```

其次，记得查看_config.ym文件中url的值，必须包含是百度站长平台注册的域名（一般有www）， 比如:

```
# URL
url: http://www.hui-wang.info
root: /
permalink: :year/:month/:day/:title/
```

最后，加入新的deployer:

```
deploy:
- type: s3 ## 这是我原来的deployer
  bucket: hui-wang.info
- type: baidu_url_submitter ## 这是新加的
```

执行hexo deploy的时候，新的连接就会被推送了。

### 原理
推送功能的实现，分为两部分：

- 新链接的产生， hexo generate 会产生一个文本文件，里面包含最新的链接
- 新链接的提交， hexo deploy 会从上述文件中读取链接，提交至百度搜索引擎

>注意：
1. 百度每天主动提交的链接数量是有限制的。
（主动推送可提交的链接数量上限是根据您提交的新产生有价值链接数量而决定的，百度会根据您提交数量的情况不定期对上限额进行调整，提交的新产生有价值链接数量越多，可提交链接的上限越高。）
2. 主动推送是否成功会在执行 hexo deploy 时显示, success后的数字为主动推送成功的链接数。

**附录**
其实，这些提交方法可以混合使用，最痛苦的是，及时提交成功了，要等好久也不知道自己的博客能否被收录。所以百度收录真的很鸡肋，比较注重网站的质量。

于是催生了另一种方案，那就是在 `Coding.net` 上进行镜像部署。这是利用 Coding.net 提供的 Coding Pages 功能另外部署一个镜像，让百度爬虫访问此镜像，普通用户还是访问位于 Github Pages 的页面。具体的这里就不在介绍，可以参考夏明额参考文章或者其他博主的文章。

**参考文章**
[Hexo 优化：提交 sitemap 及解决百度爬虫无法抓取 GitHub Pages 链接问题](http://www.yuan-ji.me/Hexo-%E4%BC%98%E5%8C%96%EF%BC%9A%E6%8F%90%E4%BA%A4sitemap%E5%8F%8A%E8%A7%A3%E5%86%B3%E7%99%BE%E5%BA%A6%E7%88%AC%E8%99%AB%E6%8A%93%E5%8F%96-GitHub-Pages-%E9%97%AE%E9%A2%98/)
[Hexo博客第三方主题next进阶教程](https://www.jianshu.com/p/1ff2fcbdd155)
[Hexo插件之百度主动提交链接](https://hui-wang.info/2016/10/23/Hexo%E6%8F%92%E4%BB%B6%E4%B9%8B%E7%99%BE%E5%BA%A6%E4%B8%BB%E5%8A%A8%E6%8F%90%E4%BA%A4%E9%93%BE%E6%8E%A5/)