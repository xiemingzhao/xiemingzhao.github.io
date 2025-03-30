---
title: Hexo博客绑定域名
categories:
  - 博客搭建
tags:
  - Hexo
copyright: true
abbrlink: Hexoblogdomain
date: 2019-05-25

---

想要小白详细版本的Github+Hexo+Next搭建博客教程，可访问我的另一篇博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/GithubHexoNextblog)。

## Hexo博客域名绑定
在我们建好了博客后，我们就可以通过`yourname.github.io`来访问你的博客，但是这个域名明显看起来不够厉害。这时候我们想的是如何将自己的博客的域名设置成像一般网址一样呢，例如我的原本的链接是[xiemingzhao.github.io](www.xiemingzhao.com)，但是经过设置绑定后将我得博客的新域名变成了[www.xiemingzhao.com](www.xiemingzhao.com)，并且打开网之后各个页面显示的连接也是www.xiemingzhao.com作为开头。接下来我们就开始。

### 1. 域名购买
域名这个东西当然不是免费的啦，毕竟网址这个东西还是需要有管控的。购买域名的地方有很多，本人是在[阿里云https://www.aliyun.com/](https://www.aliyun.com/)上面购买的，原因就不多说，大品牌值得信奈。打开阿里云官网，你需要做一下两件事情：

<!--more-->

>注册阿里云账号，也可以用淘宝或者支付宝账号登录，毕竟是一家子
然后进行实名认证，毕竟买域名还是需要正式一点，要备案的

![aliyun1](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun1.jpg)

#### **注意啦！敲黑板啦！大写加粗划重点啦！领券啦！**
>买东西没优惠怎么行，当然土豪请绕路！一般购买一个`.com`的域名，最便宜也得60左右一年，像我等底层人士还是觉得蛮贵的，那么优惠有木有，当然啦！可直接搜索`阿里云 优惠券`仔细找一找能够找得到。既然看了我的博客，肯定要对得起看官了，这里附上两个常用渠道，一个是[官方大礼包](https://promotion.aliyun.com/ntms/yunparter/invite.html?userCode=r3yteowb)，这个一般有一定限制，优惠不一定能够满足条件。另一个就是[云优惠大全](https://www.langtto.com/aliyun/54/)，这里长期更新，有各种优惠口令，我的优惠就是从这里领取的，总价300左右的单子省了好几十大洋，还是蛮不错的。

目前实名认证后的审核应该是很快的，秒速。接着你就可以在最上方点击域名，然后在搜索框里输入你想搜的域名，一般`.com`最抢手也是最贵的，而且决定价格很重要的一个因素就是域名的通识性，辨识度或者寓意于浩价格肯定越贵。得益于本人的名字较长，全拼的`.com`是基准价，于是就购买了，跟后面流程没什么好说的，主要就是填写一些个人信息包括地址等等。

### 2. 获取博客站点ip
首先使用`win+R`快捷键打开“运行”窗口，输入cmd运行调出命令行控制台。输入以下命令行，查询你自己博客站点的ip：
```
ping yourname.github.io
```
注意上述代码中的`yourname`要换成你自己的哦。如下图，即使静秋超市也没关系，只需要第一行信息，你就可以找到你自己博客站点对应的ip了，这个在后续的域名绑定中有用。

![ping blogip](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun2.jpg)

### 3. 域名解析配置
我们接着第一步购买域名后的步骤，回到首页点击用户信息旁边的`控制台`：

![aliyun home](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun3.jpg)

再点击`域名`选项，当然你通过其他渠道进入到这里也行：

![aliyun cmd](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun4.jpg)

接下来，我们找到你购买的域名，点击后面的解析，进入域名解析页面：

![域名解析](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun5.jpg)

然后删除默认的，添加如下两条解析配置，最好按照我这里的配置来，因为亲测在后续的站点收录等地方这么设置最好不会出bug。

![解析配置](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun6.jpg)

### 4. 站点域名绑定
在解析配置好了之后，登录你自己的`github`，进入到博客站点对应的仓库，进入`settings`，找到如下图的`custom domain`配置区域，填写你购买的域名，这里加不加`www.`都是阔以的，我是加了的，并且勾选上`Enforce HTTPS`选项，勾选这个在以后其他地方有用，有利于收录。

![绑定站点域名](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun7.jpg)

### 5. 解析文件配置
这是最后一步了，进入你本地博客站点根目录下的`/source/`文件夹中，创建一个名为`CNAME`的文件，没有错，不带任何后缀名，里面只需要协商你刚才购买的域名就可以了。这里面填写的域名加不加`www.`会有不同的效果：
>a. 如果你填写的是没有www的，比如 xiemingzhao.com，那么无论是访问 https://www.xiemingzhao.com 还是 http://xiemingzhao.me ，都会自动跳转到 http://xiemingzhao.com。
b. 如果你填写的是带www的，比如 www.xiemingzhao.com ，那么无论是访问 http://www.xiemingzhao.com 还是 http://xiemingzhao.com ，都会自动跳转到 http://www.xiemingzhao.com。
c. 如果你填写的是其它子域名，比如 abc.xiemingzhao.com，那么访问 http://abc.xiemingzhao.com 没问题，但是访问 http://xiemingzhao.com ，不会自动跳转到 http://abc.xiemingzhao.com。

![CNAME](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/aliyun8.jpg)

**另外，在绑定了这个新的域名后，原来的`yourname.github.io`并没有失效哦，会自动解析到新域名上，是不是很酷！**

最后在站点配置文件`_config.yml`文件中将站点的url改成自己购买的新域名即可，例如下方我自己的配置：
```
# URL
## If your site is put in a subdirectory, set url as 'http://yoursite.com/child' and root as '/child/'
url: http://www.xiemingzhao.com
root: /
permalink: :year/:month/:day/:title/
permalink_defaults:
```

到这里就结束了，使用三个经典的发布命令后于就可以测试一下了。

**参考博文**
[Hexo个人博客域名绑定 简明教程（小白篇）](https://www.jianshu.com/p/e3169b681038)

---

