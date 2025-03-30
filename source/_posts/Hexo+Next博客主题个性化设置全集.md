---
title: Hexo+Next博客主题个性化设置全集
categories:
  - 博客搭建
tags:
  - Hexo
  - Next
copyright: true
abbrlink: HexoNextindividual
date: 2019-06-10

---

想要小白详细版本的Github+Hexo+Next搭建博客教程，可访问我的另一篇博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/GithubHexoNextblog)。

**注意：以下非特殊说明路径都是基于你本地博客的根目录，效果主要基于hexo+next实现效果，大部分效果均可在[我的博客](www.xiemingzhao.com)中保留，可先睹为快，再决定是否需要**

## 1.实现展示*fork me on github*效果
先上效果图：

![fork me on github](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual1.jpg)

<!--more-->

这类效果图主要有两类样式，分别是显示`fork me on github`还有一个显示github图标的，可分别在[第一种地址](https://github.blog/2008-12-19-github-ribbons/)和[第二种地址](http://tholman.com/github-corners/)里面选择自己喜欢的款式，然后复制对应框中的代码。接着打开`themes/next/layout/_layout.swig`文件，将刚才复制的代码粘贴在`<div class="headband"></div>`代码下面，保持缩进与其对其，同时把`herf`后面对应的github地址换成你自己的。这里需要注意的是，目前复制的代码中新增了width和height参数，却没有了位置的参数。如下图所示，如果你想跟我一样只设置位置，大小默认，那么就可以删除width和height参数，并新增top,right,和border参数。想要测试在线效果的话，可用`hexo s`本地测试，或者`hexo d -g`线上测试。

![fork code](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual2.jpg)

## 2.添加动态背景
先看效果图：

![background](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual3.gif)

**注意：**如果你是使用`next`主题并且版本在5.1.1以上，例如我自己就是，那么设置起来就极其简单，直接在主题配置文件中找到`canvas_nest`配置字段改成true即可：
```
canvas_nest: true
```

那如果你符合next以及5.1.1版本的要求怎么办呢？不着急，我们通过一些配置文件的修改也能够达到目的。上述符合的话只需要打开开关的原因是以下这些配置都已经集合进去了。

### 修改`_layout.swig`
首先我们打开配置文件：`/themes/next/layout/_layout.swig`，然后在 `< /body>`之前添加代码(注意不要放在`< /head>`的后面)：

```
{ % if theme.canvas_nest % }
<script type="text/javascript"
color="0,0,0" opacity='0.5' zIndex="-1" count="150" src="//cdn.bootcss.com/canvas-nest.js/1.0.0/canvas-nest.min.js"></script>
{ % endif % }
```

其中有几个配置参数解释一下：

```
color： 线条的颜色，默认是 0,0,0 ，对应的是(R,G,B)参数；
opacity： 线条的透明度（0~1），默认是0.5；
count： 线条的总数量，默认是150；
zIndex： 北京的z-index属性，css属性用户控制所在层的位置，默认是-1。
```

### 修改主题配置文件
我们打开主题配置文件：`/themes/next/_config.yml`，在里面添加以下代码，一版放在最后面即可：

```
# --------------------------------------------------------------
# background settings
# --------------------------------------------------------------
# add canvas-nest effect
# see detail from https://github.com/hustcc/canvas-nest.js
canvas_nest: true
```

然后就可以使用经典的三个不输命令来测试了，在gitbash中运行`hexo clean`和`hexo g`，以及`hexo s`可在本地测试，或者`hexo d`即可在线上进行展示效果。

## 3. 点击出现桃心效果
先看效果图：

![click](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual4.gif)

首先打开目录：`/themes/next/source/js/src/`，之后在里面新建一个`love.js`的文件，文件里面复制进去以下内容：

```
!function(e,t,a){function n(){c(".heart{width: 10px;height: 10px;position: fixed;background: #f00;transform: rotate(45deg);-webkit-transform: rotate(45deg);-moz-transform: rotate(45deg);}.heart:after,.heart:before{content: '';width: inherit;height: inherit;background: inherit;border-radius: 50%;-webkit-border-radius: 500%;-moz-border-radius: 50%;position: fixed;}.heart:after{top: -5px;}.heart:before{left: -5px;}"),o(),r()}function r(){for(var e=0;e<d.length;e++)d[e].alpha<=0?(t.body.removeChild(d[e].el),d.splice(e,1)):(d[e].y--,d[e].scale+=.004,d[e].alpha-=.013,d[e].el.style.cssText="left:"+d[e].x+"px;top:"+d[e].y+"px;opacity:"+d[e].alpha+";transform:scale("+d[e].scale+","+d[e].scale+") rotate(45deg);background:"+d[e].color+";z-index:99999");requestAnimationFrame(r)}function o(){var t="function"==typeof e.onclick&&e.onclick;e.onclick=function(e){t&&t(),i(e)} }function i(e){var a=t.createElement("div");a.className="heart",d.push({el:a,x:e.clientX-5,y:e.clientY-5,scale:1,alpha:1,color:s()}),t.body.appendChild(a)}function c(e){var a=t.createElement("style");a.type="text/css";try{a.appendChild(t.createTextNode(e))}catch(t){a.styleSheet.cssText=e}t.getElementsByTagName("head")[0].appendChild(a)}function s(){return"rgb("+~~(255*Math.random())+","+~~(255*Math.random())+","+~~(255*Math.random())+")"}var d=[];e.requestAnimationFrame=function(){return e.requestAnimationFrame||e.webkitRequestAnimationFrame||e.mozRequestAnimationFrame||e.oRequestAnimationFrame||e.msRequestAnimationFrame||function(e){setTimeout(e,1e3/60)} }(),n()}(window,document);
```

接着打开配置文件`/themes/next/layout/_layout.swig`，在最后加上以下内容：

```
<!-- 页面点击小红心 -->
<script type="text/javascript" src="/js/src/love.js"></script>
```

到此就完成了，可以部署测试一下效果。

## 4. 修改文章链接样式
先看效果图：

![blog link style](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual5.gif)

打开配置文件：`themes/next/source/css/_common/components/post/post.styl`，在最后面添加如下的代码来设置：

```
// 文章内链接文本样式
.post-body p a{
  color: #0593d3;
  border-bottom: none;
  border-bottom: 1px solid #0593d3;
  &:hover {
    color: #fc6423;
    border-bottom: none;
    border-bottom: 1px solid #fc6423;
  }
}
```

其中`.post-body`是为了值效果不影响标题，而后面的`p`是为了起到不影响首页`阅读原文`的效果，`color: #fc6423;`是颜色的配置项，可以自定义。

## 5. 修改文章底部标签符号**#**
默认的文章底部显示的标签是以`#`来展示的，看上去蛮怪的，我们改成另一种样式，

我们打开模板配置文件：`/themes/next/layout/_macro/post.swig`，然后搜索`rel="tag">#`，将其中的`#`换成`<i class="fa fa-tag"></i>`，保存后部署即可。

## 6. 文章末尾添加“结束”标记
先看效果图：

![blog ending](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual6.jpg)

首先打开文件夹：`/themes/next/layout/_macro`，然后在其中新建`passage-end-tag.swig`文件，并添加以下内容：

```
<div>
    { % if not is_index % }
        <div style="text-align:center;color: #ccc;font-size:14px;">-------------本文结束<i class="fa fa-paw"></i>感谢您的阅读-------------</div>
    { % endif % }
</div>
```

其中`-------------本文结束`和`感谢您的阅读-------------`两处可根据自己的喜好进行修改。

接着，打开配置文件：`/themes/next/layout/_macro/post.swig`，在`END POST BODY`和`<footer class="post-footer">`之间添加以下代码，一般就放在紧邻`END POST BODY`后面即可：

```
<div>
  { % if not is_index % }
    { % include 'passage-end-tag.swig' % }
  { % endif % }
</div>
```

最后打开主题配置文件，在最后面添加以下代码即可：
```
# 文章末尾添加“本文结束”标记
passage_end_tag:
  enabled: true
```

到这里就完成了，可以部署后看看线上的效果，每一篇文章都添加了。

## 7. 头像触碰旋转
先上效果图：

![headpicture](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual7.gif)

头像的配置这里不赘述了，可参考本文开头提到的另一篇博客。接下来为了实现触碰旋转，我们需要先打开配置文件：`/themes/next/source/css/_common/components/sidebar/sidebar-author.styl`，然后再最后添加如下一段代码：

```
.site-author-image {
  display: block;
  margin: 0 auto;
  padding: $site-author-image-padding;
  max-width: $site-author-image-width;
  height: $site-author-image-height;
  border: $site-author-image-border-width solid $site-author-image-border-color;

  /* 头像圆形 */
  border-radius: 80px;
  -webkit-border-radius: 80px;
  -moz-border-radius: 80px;
  box-shadow: inset 0 -1px 0 #333sf;

  /* 设置循环动画 [animation: (play)动画名称 (2s)动画播放时长单位秒或微秒 (ase-out)动画播放的速度曲线为以低速结束 
    (1s)等待1秒然后开始动画 (1)动画播放次数(infinite为循环播放) ]*/
 

  /* 鼠标经过头像旋转360度 */
  -webkit-transition: -webkit-transform 1.0s ease-out;
  -moz-transition: -moz-transform 1.0s ease-out;
  transition: transform 1.0s ease-out;
}

img:hover {
  /* 鼠标经过停止头像旋转 
  -webkit-animation-play-state:paused;
  animation-play-state:paused;*/

  /* 鼠标经过头像旋转360度 */
  -webkit-transform: rotateZ(360deg);
  -moz-transform: rotateZ(360deg);
  transform: rotateZ(360deg);
}

/* Z 轴旋转动画 */
@-webkit-keyframes play {
  0% {
    -webkit-transform: rotateZ(0deg);
  }
  100% {
    -webkit-transform: rotateZ(-360deg);
  }
}
@-moz-keyframes play {
  0% {
    -moz-transform: rotateZ(0deg);
  }
  100% {
    -moz-transform: rotateZ(-360deg);
  }
}
@keyframes play {
  0% {
    transform: rotateZ(0deg);
  }
  100% {
    transform: rotateZ(-360deg);
  }
}
```

## 8. 修改``代码块样式
先看效果图：

![code style](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual8.jpg)

首先打开配置文件：`/themes/next/source/css/_custom/custom.styl`，如果对应路径下没有的话就新建一个，然后在里面添加以下代码：

```
// Custom styles.
code {
    color: #ff7600;
    background: #fbf7f8;
    margin: 2px;
}
// 大代码块的自定义样式
.highlight, pre {
    margin: 5px 0;
    padding: 5px;
    border-radius: 3px;
}
.highlight, code, pre {
    border: 1px solid #d6d6d6;
}
```

## 9. 主页文章添加阴影效果
先看效果图：

![home paper back](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual9.jpg)

紧接着上面的步骤，继续在配置文件`/themes/next/source/css/_custom/custom.styl`中最后面添加下面的代码即可：

```
// 主页文章添加阴影效果
 .post {
   margin-top: 60px;
   margin-bottom: 60px;
   padding: 25px;
   -webkit-box-shadow: 0 0 5px rgba(202, 203, 203, .5);
   -moz-box-shadow: 0 0 5px rgba(202, 203, 204, .5);
  }
```

## 10. 侧栏社交连接
效果如下：

![social link](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual10.jpg)

首先打开主题配置文件`_config.yml`，然后搜索`social`关键词，找到`social`和`social_icons`参数配置区域，可以看到如下的代码配置块：

```
social:
  GitHub: https://github.com/xiemingzhao
  Google: https://plus.google.com/xiemingzhao
  Twitter: https://twitter.com/xiemingzhao
  #FB Page: https://www.facebook.com/yourname || facebook
  #VK Group: https://vk.com/yourname || vk
  #StackOverflow: https://stackoverflow.com/yourname || stack-overflow
  #YouTube: https://youtube.com/yourname || youtube
  #Instagram: https://instagram.com/yourname || instagram
  #Skype: skype:yourname?call|chat || skype

social_icons:
  enable: true
  icons_only: false
  transition: false
  GitHub: github
  Google: google
  Twitter: twitter
```

如上所示，`social`模块是配置各个社交连接项的，不需要的可以注释掉，也可以新增其他的社交项，名称也都没有限制，社交项对应的图标是使用`fontawesome`的icon图标的，可以在[fontawesome图标官网](https://fontawesome.com/icons?from=io)找到你喜欢的。图标的配置可以在`social`模块中的对应链接后面使用`||`隔开配置，也可在下面的`social_icons`模块配置，不管怎样，最后记得要把`social_icons`模块中的`enable`设为`true`哦。

## 11. 站点底部和文章顶部显示访问量
先看看如下两个效果图：

![uv and pv](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual11.jpg)

![site uv and pv](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual12.jpg)

使用next主题的好处在这里就体现出来了，配置这个很简单，只需要打开主题配置文件`/theme/next/_config.yml`后，搜索`busuanzi`找到如下配置代码块后进行相关配置即可：

```
# Show PV/UV of the website/page with busuanzi.
# Get more information on http://ibruce.info/2015/04/04/busuanzi/
busuanzi_count:
  # count values only if the other configs are false
  enable: true
  # custom uv span for the whole site
  site_uv: true
  site_uv_header: <i class="fa fa-user"></i> 访客数
  site_uv_footer: 人
  # custom pv span for the whole site
  site_pv: true
  site_pv_header: <i class="fa fa-eye"></i> 总访问量
  site_pv_footer: 次
  # custom pv span for one page only
  page_pv: true
  page_pv_header: <i class="fa fa-file-o"></i> 阅读数
  page_pv_footer: 次
```

如上代码所示的是我本人的配置，主要分为三大块：

>site_uv表示是否显示整个网站的UV数
site_pv表示是否显示整个网站的PV数
page_pv表示是否显示每个页面的PV数

需要的则把对应的开关配成`true`，另`site_uv_header`配置项中的文字是显示的名称，而`site_pv_footer`则是对应的统计单位。当然如果你是其他主题，也可配置，网上教程很多。到此就配置结束，可以使以部署测试效果，但是这里如果使用`hexo s`来测试，你会发现数字特别大，这是正常现象，因为不蒜子用户使用一个存储空间，所以你只要`hexo d -g`部署到线上进行测试就完毕了。

## 12. 网站底部字数统计
如上述第11部分效果图所示。

首先需要安装一个统计包，我们将控制台路径切到根目录或者在根目录调出`git bash`，然后运行如下代码：

```
npm install hexo-wordcount --save
```

接着打开配置文件`/themes/next/layout/_partials/footer.swig`，之后在最下面新增以下代码即可：

```
<div class="theme-info">
  <div class="powered-by"></div>
  <span class="post-count">博客全站共{ { totalcount(site) } }字</span>
</div>
```

## 13. 文章页面的统计功能
如上述第11部分效果图所示。

这一块的统计主要是对于每篇文章在头部显示字数和推荐阅读时长，配置方法与上述类似，现在根目录运行以下代码安装统计包：

```
npm install hexo-wordcount --save
```

紧接着在主题配置文件中搜索`post_wordcount`配置模块，并按照如下代码进行配置，如果没有的话直接新增也可以：

```
# Post wordcount display settings
# Dependencies: https://github.com/willin/hexo-wordcount
post_wordcount:
  item_text: true
  wordcount: true
  min2read: true
```

## 14. 顶部加载进度条
先看效果图：

![loading](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual13.gif)

这个功能在next主题中配置起来又是如此的简单，只需要打开主题配置文件，之后搜搜`pace`找到如下的代码配置模块：

```
# Progress bar in the top during page loading.
pace: true
# Themes list:
#pace-theme-big-counter
#pace-theme-bounce
#pace-theme-barber-shop
#pace-theme-center-atom
#pace-theme-center-circle
#pace-theme-center-radar
#pace-theme-center-simple
#pace-theme-corner-indicator
#pace-theme-fill-left
#pace-theme-flash
#pace-theme-loading-bar
#pace-theme-mac-osx
#pace-theme-minimal
# For example
# pace_theme: pace-theme-center-simple
pace_theme: pace-theme-minimal
```

将其中的`pace`项设为`true`即可打开进度条效果，其中`pace_theme`配置项是选择进度条效果，可选的配置项都列在了各个注释中，可以参考[这篇文章](https://www.jianshu.com/p/d08513d38786)看看不同进度条的展示效果，选择一个自己最喜欢的即可。

## 15. 添加`README.md`文件
我们在 Hexo 目录下的 source 根目录下添加一个 README.md 文件，修改站点配置文件 _config.yml，将 skip_render 参数的值设置为：`skip_render: README.md`即可。

这样再每次部署的时候就不会重新渲染`README.md`文件，否则每次都要修改对应的文件。

## 16. 修改网站图标
图标展示如下所示：

![site picture](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual14.jpg)

Hexo博客默认图标是一个六边形的黑色背景白色前景N的图案。如果你想要修改对应的图案或者从其他渠道也可以，可以去[EasyIcon](http://www.easyicon.net/)找到喜欢的图案下载，下载的时候有好几种大小，该如何选择呢。

我们可以进入`/themes/next/source/images`目录，并且同时打开主题配置文件，搜索`favicon`找到其如下的配置代码模块：

```
# For example, you put your favicons into `hexo-site/source/images` directory.
# Then need to rename & redefine they on any other names, otherwise icons from Next will rewrite your custom icons in Hexo.
favicon:
  small: /images/favicon-16x16-next.png
  medium: /images/favicon-32x32-next.png
  apple_touch_icon: /images/apple-touch-icon-next.png
  safari_pinned_tab: /images/logo.svg
  android_manifest: /images/manifest.json
  ms_browserconfig: /images/browserconfig.xml
```

对比images文件夹里的文件，相比你就知道各种大小的 favicon 图案对应什么场景需求了，然后将你选择好的不同大小的图案替换现有的图案，保持名称和后缀一样就可以了。

## 16. 增加版权信息
先看效果图：

![copyright](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual15.jpg)

对文章添加版本信息也是对自己发布的内容的一种保护方式，也是为了减少别人引用时候的费力度，于人于己都是好事。我门首先打开目录：`themes/next/layout/_macro/`，然后新建一个文件`my-copyright.swig`，内容添加如下代码：

```
{ % if page.copyright % }
<div class="my_post_copyright">
  <script src="//cdn.bootcss.com/clipboard.js/1.5.10/clipboard.min.js"></script>
  
  <!-- JS库 sweetalert 可修改路径 -->
  <script src="https://cdn.bootcss.com/jquery/2.0.0/jquery.min.js"></script>
  <script src="https://unpkg.com/sweetalert/dist/sweetalert.min.js"></script>
  <p><span>本文标题:</span><a href="{ { url_for(page.path) } }">{ { page.title } }</a></p>
  <p><span>文章作者:</span><a href="/" title="访问 { { theme.author } } 的个人博客">{ { theme.author } }</a></p>
  <p><span>发布时间:</span>{ { page.date.format("YYYY年MM月DD日 - HH:MM") } }</p>
  <p><span>最后更新:</span>{ { page.updated.format("YYYY年MM月DD日 - HH:MM") } }</p>
  <p><span>原始链接:</span><a href="{ { url_for(page.path) } }" title="{ { page.title } }">{ { page.permalink } }</a>
    <span class="copy-path"  title="点击复制文章链接"><i class="fa fa-clipboard" data-clipboard-text="{ { page.permalink } }"  aria-label="复制成功！"></i></span>
  </p>
  <p><span>许可协议:</span><i class="fa fa-creative-commons"></i> <a rel="license" href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" title="Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)">署名-非商业性使用-禁止演绎 4.0 国际</a> 转载请保留原文链接及作者。</p>  
</div>
<script> 
    var clipboard = new Clipboard('.fa-clipboard');
      $(".fa-clipboard").click(function(){
      clipboard.on('success', function(){
        swal({   
          title: "",   
          text: '复制成功',
          icon: "success", 
          showConfirmButton: true
          });
        });
    });  
</script>
{ % endif % }
```

可以看到代码中配置了主要会披露的几项内容：
>1.本文标题
2.文章作者
3.发布时间
4.最后更新
5.原始链接
6.许可协议

根据你的需求选择去留就好，在我们设置里面剔除了发布时间和最后更新的披露。

接下来我们再打开目录：`next/source/css/_common/components/post/`，并在其中新建文件：`my-post-copyright.styl`，之后里面添加如下代码：

```
.my_post_copyright {
  width: 85%;
  max-width: 45em;
  margin: 2.8em auto 0;
  padding: 0.5em 1.0em;
  border: 1px solid #d3d3d3;
  font-size: 0.93rem;
  line-height: 1.6em;
  word-break: break-all;
  background: rgba(255,255,255,0.4);
}
.my_post_copyright p{margin:0;}
.my_post_copyright span {
  display: inline-block;
  width: 5.2em;
  color: #b5b5b5;
  font-weight: bold;
}
.my_post_copyright .raw {
  margin-left: 1em;
  width: 5em;
}
.my_post_copyright a {
  color: #808080;
  border-bottom:0;
}
.my_post_copyright a:hover {
  color: #a3d2a3;
  text-decoration: underline;
}
.my_post_copyright:hover .fa-clipboard {
  color: #000;
}
.my_post_copyright .post-url:hover {
  font-weight: normal;
}
.my_post_copyright .copy-path {
  margin-left: 1em;
  width: 1em;
  +mobile(){display:none;}
}
.my_post_copyright .copy-path:hover {
  color: #808080;
  cursor: pointer;
}
```

还没结束呢，我们还需要打开配置文件：`/themes/next/layout/_macro/post.swig`，搜索找到`next/layout/_macro/post.swig`所在的代码块，在其前面新增一段代码，最终结果如下所示：

```
    <div>
      { % if not is_index % }
      { % include 'passage-end-tag.swig' % }
      { % endif % }
    </div>

    <div>
      { % if not is_index % }
        { % include 'my-copyright.swig' % }
      { % endif % }
    </div>
```

要保持对其哦，然后就差最后一步了，打开配置文件：`/themes/next/source/css/_common/components/post/post.styl`，再最后新增一行如下的代码：

```
@import "my-post-copyright"
```

保存后部署即可。可能这时候你发现文章还是没有出现版权，那是因为还有一个很关键的因素，那就是你需要在你想添加版权的文章的头部增加`copyright: true`的配置项。例如本篇博文的配置汇总就有：

```
---
title: Hexo+Next博客主题个性化设置全集
date: 2019-06-10
abbrlink: Hexo+Next_individual
categories:
- 博客搭建
tags:
- Hexo
- Next
copyright: true
---
```

如果不想每次手动添加，想要实现`hexo new`的时候就自动添加的话，也是可以的。还记得之前提过新建博客的配置模板文件吗？打开`/scaffolds/post.md`在其中添加`copyright: true`即可。

## 17. 隐藏或修改底部`驱动/主题`信息
默认的话底部一般会显示`由 Hexo 强力驱动 | Next 主题`之类的信息。如果你想要去掉或者删除，则可以打开配置文件`/themes/next/layout/_partials/footer.swig/`，之后找到如下两块代码：

```
{ % if theme.footer.powered % }
  <div class="powered-by">{#
  #}{ { __('footer.powered', '<a class="theme-link" target="_blank" href="https://hexo.io">Hexo</a>') } }{#
#}</div>
{ % endif % }

{ % if theme.footer.theme.enable % }
  <div class="theme-info">{#
  #}{ { __('footer.theme') } } &mdash; {#
  #}<a class="theme-link" target="_blank" href="https://github.com/iissnan/hexo-theme-next">{#
    #}NexT.{ { theme.scheme } }{#
  #}</a>{ % if theme.footer.theme.version % } v{ { theme.version } }{ % endif % }{#
#}</div>
{ % endif % }
```

简单看一下，第一部分包含`powered-by`关键词就知道是设置`由 Hexo 强力驱动`代码块，另一部分有关键词`theme-info`则是设置主题信息的。根据自己需要进行修改和删除即可。

## 18. 修改底部桃心标志
这个比较简单，打开文件：`/themes/next/layout/_partials/footer.swig`，找到以下代码：

```
  <span class="with-love">
    <i class="fa fa-{ { theme.footer.icon } }"></i>
  </span>
```

其中`<i class="fa fa-{ { theme.footer.icon } }"></i>`就是这个图标的配置，同样的可以去[icons图库](http://fontawesome.io/icons/)找到自己喜欢的图标替换这里的配置即可。然鹅，我自己没有修改，是因为没找到比这个心更合适的，当然教程还是要写给大家的。

## 19. 文章加密访问
虽然博客主要是对外交流使用，但是有时候也可以用来记录自己的一些私密事情，这时候难免像有个只有自己能进去的一片天地。毕竟以前的qq空间还有加密功能呢。Hexo自然也少不了，下面我们就开始。

首先打开配置文件：`/themes/next/layout/_partials/head.swig`，在`<meta name`代码坐在模块和`{ % if theme.pace % }`所在代码块之间新增如下代码即可：

```
<meta charset="UTF-8"/>
<meta http-equiv="X-UA-Compatible" content="IE=edge" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<meta name="theme-color" content="{ { theme.android_chrome_color } }">

<script>
    (function(){
        if('{ { page.password } }'){
            if (prompt('请输入文章密码') !== '{ { page.password } }'){
                alert('密码错误！');
                history.back();
            }
        }
    })();
</script>

{ % if theme.pace % }
  { % set pace_css_uri = url_for(theme.vendors._internal + '/pace/'+ theme.pace_theme +'.min.css?v=1.0.2') % }
```

同样的还需要主动触发的一步，也就是需要在你想要加密的文章头部添加`password:xxxxxx`配置项即可，后面的`xxxxxx`即是这篇文章的查看密码。

## 20. 添加分享功能
先看效果图：

![share](https://mzxie-image.oss-cn-hangzhou.aliyuncs.com/hexoBlog/next-individual16.jpg)

Hexo可以给博客增加分享功能，这样就可以实现分享页面/图片以及文章了。有的人使用`jiathis`分享功能，直接在主题配置文件中找到这个模块，将其改为`true`即可，但是亲测好像不是特别好用。

这里介绍的是百度分享功能，简洁实用。首先打开站点配置文件，搜索`baidushare`关键词，如下配置打开分享功能：

```
# Baidu Share
# Available value:
#    button | slide
# Warning: Baidu Share does not support https.
baidushare:
  type: button
  baidushare: true
```

但是在部署使用的时候可能会引发`Warning: Baidu Share does not support https`从而不可使用。不怕，博主[hrwhisper](https://github.com/hrwhisper/baiduSh)发布了一个修复方案，我这里简介一下。

首先下载上面博文发布的[static](https://github.com/hrwhisper/baiduShare)文件夹。下载后解压，将`static`文件夹保存在`/themes/next/source`文件夹下面。

最后打开文件：`/themes/next/layout/_partials/share/baidushare.swig`，将文件末尾的一行代码按如下提示进行修改：

```
.src='http://bdimg.share.baidu.com/static/api/js/share.js?v=89860593.js?cdnversion='+~(-new Date()/36e5)];
#改为
.src='/static/api/js/share.js?v=89860593.js?cdnversion='+~(-new Date()/36e5)];
```

修改完成后，重新部署就可以正常使用分享功能了。

## 21. 博文置顶
有时候我们想将某一篇文章置顶在首页。我们可以这么实现。修改 `hero-generator-index` 插件，把文件`node_modules/hexo-generator-index/lib/generator.js` 内的代码替换为：

```
'use strict';
var pagination = require('hexo-pagination');
module.exports = function(locals){
  var config = this.config;
  var posts = locals.posts;
    posts.data = posts.data.sort(function(a, b) {
        if(a.top && b.top) { // 两篇文章top都有定义
            if(a.top == b.top) return b.date - a.date; // 若top值一样则按照文章日期降序排
            else return b.top - a.top; // 否则按照top值降序排
        }
        else if(a.top && !b.top) { // 以下是只有一篇文章top有定义，那么将有top的排在前面（这里用异或操作居然不行233）
            return -1;
        }
        else if(!a.top && b.top) {
            return 1;
        }
        else return b.date - a.date; // 都没定义按照文章日期降序排
    });
  var paginationDir = config.pagination_dir || 'page';
  return pagination('', posts, {
    perPage: config.index_generator.per_page,
    layout: ['index', 'archive'],
    format: paginationDir + '/%d/',
    data: {
      __index: true
    }
  });
};
```

之后，想要指定某篇文章置顶的时候，就可以在文章的头部配置`top: 100`，top值越大越靠前。其实就是文章优先按照top值进行降序排列，没有top值的或者top值相同的就按照文章时间进行降序排列。

## 22. 修改字体大小和鼠标样式
字体大小的修改，可以打开文件： `/themes/next/source/css/ _variables/base.styl`，将`$font-size-base`配置项改成你想要的大小即可，例如：

```
$font-size-base =16px
```

鼠标样式的修改则需要打开文件： `/themes/next/source/css/_custom/custom.styl`，然后在里面添加如下代码：

```
// 鼠标样式
  * {
      cursor: url("http://om8u46rmb.bkt.clouddn.com/sword2.ico"),auto!important
  }
  :active {
      cursor: url("http://om8u46rmb.bkt.clouddn.com/sword1.ico"),auto!important
  }
```

其中`url` 里面必须是 ico 图片，ico 图片可以上传到图床，生成对应外链添加到这里就好了。

## 23. 点击爆炸效果
在前我们介绍了如何实现点击出现`心`的效果，这里介绍一个更浮夸的效果，点击时候爆炸。配置方式差不多的，首先在目录： `/themes/next/source/js/src` 里面建一个叫 `fireworks.js` 的文件，添加如下代码：

```
"use strict";function updateCoords(e){pointerX=(e.clientX||e.touches[0].clientX)-canvasEl.getBoundingClientRect().left,pointerY=e.clientY||e.touches[0].clientY-canvasEl.getBoundingClientRect().top}function setParticuleDirection(e){var t=anime.random(0,360)*Math.PI/180,a=anime.random(50,180),n=[-1,1][anime.random(0,1)]*a;return{x:e.x+n*Math.cos(t),y:e.y+n*Math.sin(t)} }function createParticule(e,t){var a={};return a.x=e,a.y=t,a.color=colors[anime.random(0,colors.length-1)],a.radius=anime.random(16,32),a.endPos=setParticuleDirection(a),a.draw=function(){ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.fillStyle=a.color,ctx.fill()},a}function createCircle(e,t){var a={};return a.x=e,a.y=t,a.color="#F00",a.radius=0.1,a.alpha=0.5,a.lineWidth=6,a.draw=function(){ctx.globalAlpha=a.alpha,ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.lineWidth=a.lineWidth,ctx.strokeStyle=a.color,ctx.stroke(),ctx.globalAlpha=1},a}function renderParticule(e){for(var t=0;t<e.animatables.length;t++){e.animatables[t].target.draw()} }function animateParticules(e,t){for(var a=createCircle(e,t),n=[],i=0;i<numberOfParticules;i++){n.push(createParticule(e,t))}anime.timeline().add({targets:n,x:function(e){return e.endPos.x},y:function(e){return e.endPos.y},radius:0.1,duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule}).add({targets:a,radius:anime.random(80,160),lineWidth:0,alpha:{value:0,easing:"linear",duration:anime.random(600,800)},duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule,offset:0})}function debounce(e,t){var a;return function(){var n=this,i=arguments;clearTimeout(a),a=setTimeout(function(){e.apply(n,i)},t)} }var canvasEl=document.querySelector(".fireworks");if(canvasEl){var ctx=canvasEl.getContext("2d"),numberOfParticules=30,pointerX=0,pointerY=0,tap="mousedown",colors=["#FF1461","#18FF92","#5A87FF","#FBF38C"],setCanvasSize=debounce(function(){canvasEl.width=2*window.innerWidth,canvasEl.height=2*window.innerHeight,canvasEl.style.width=window.innerWidth+"px",canvasEl.style.height=window.innerHeight+"px",canvasEl.getContext("2d").scale(2,2)},500),render=anime({duration:1/0,update:function(){ctx.clearRect(0,0,canvasEl.width,canvasEl.height)} });document.addEventListener(tap,function(e){"sidebar"!==e.target.id&&"toggle-sidebar"!==e.target.id&&"A"!==e.target.nodeName&&"IMG"!==e.target.nodeName&&(render.play(),updateCoords(e),animateParticules(pointerX,pointerY))},!1),setCanvasSize(),window.addEventListener("resize",setCanvasSize,!1)}"use strict";function updateCoords(e){pointerX=(e.clientX||e.touches[0].clientX)-canvasEl.getBoundingClientRect().left,pointerY=e.clientY||e.touches[0].clientY-canvasEl.getBoundingClientRect().top}function setParticuleDirection(e){var t=anime.random(0,360)*Math.PI/180,a=anime.random(50,180),n=[-1,1][anime.random(0,1)]*a;return{x:e.x+n*Math.cos(t),y:e.y+n*Math.sin(t)} }function createParticule(e,t){var a={};return a.x=e,a.y=t,a.color=colors[anime.random(0,colors.length-1)],a.radius=anime.random(16,32),a.endPos=setParticuleDirection(a),a.draw=function(){ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.fillStyle=a.color,ctx.fill()},a}function createCircle(e,t){var a={};return a.x=e,a.y=t,a.color="#F00",a.radius=0.1,a.alpha=0.5,a.lineWidth=6,a.draw=function(){ctx.globalAlpha=a.alpha,ctx.beginPath(),ctx.arc(a.x,a.y,a.radius,0,2*Math.PI,!0),ctx.lineWidth=a.lineWidth,ctx.strokeStyle=a.color,ctx.stroke(),ctx.globalAlpha=1},a}function renderParticule(e){for(var t=0;t<e.animatables.length;t++){e.animatables[t].target.draw()} }function animateParticules(e,t){for(var a=createCircle(e,t),n=[],i=0;i<numberOfParticules;i++){n.push(createParticule(e,t))}anime.timeline().add({targets:n,x:function(e){return e.endPos.x},y:function(e){return e.endPos.y},radius:0.1,duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule}).add({targets:a,radius:anime.random(80,160),lineWidth:0,alpha:{value:0,easing:"linear",duration:anime.random(600,800)},duration:anime.random(1200,1800),easing:"easeOutExpo",update:renderParticule,offset:0})}function debounce(e,t){var a;return function(){var n=this,i=arguments;clearTimeout(a),a=setTimeout(function(){e.apply(n,i)},t)} }var canvasEl=document.querySelector(".fireworks");if(canvasEl){var ctx=canvasEl.getContext("2d"),numberOfParticules=30,pointerX=0,pointerY=0,tap="mousedown",colors=["#FF1461","#18FF92","#5A87FF","#FBF38C"],setCanvasSize=debounce(function(){canvasEl.width=2*window.innerWidth,canvasEl.height=2*window.innerHeight,canvasEl.style.width=window.innerWidth+"px",canvasEl.style.height=window.innerHeight+"px",canvasEl.getContext("2d").scale(2,2)},500),render=anime({duration:1/0,update:function(){ctx.clearRect(0,0,canvasEl.width,canvasEl.height)} });document.addEventListener(tap,function(e){"sidebar"!==e.target.id&&"toggle-sidebar"!==e.target.id&&"A"!==e.target.nodeName&&"IMG"!==e.target.nodeName&&(render.play(),updateCoords(e),animateParticules(pointerX,pointerY))},!1),setCanvasSize(),window.addEventListener("resize",setCanvasSize,!1)};
```

紧接着，打开文件： `/themes/next/layout/_layout.swig`，在 `</body>`上面添加如下代码：

```
{ % if theme.fireworks % }
   <canvas class="fireworks" style="position: fixed;left: 0;top: 0;z-index: 1; pointer-events: none;" ></canvas> 
   <script type="text/javascript" src="//cdn.bootcss.com/animejs/2.2.0/anime.min.js"></script> 
   <script type="text/javascript" src="/js/src/fireworks.js"></script>
{ % endif % }
```

最后进入主题配置文件，在里面最后按如下配置开启效果即可：

```
# Fireworks
fireworks: true
```

然鹅，这里我再次没有选择，因为这个效果感觉很容易影响阅读，华而不实。

**参考博文**
[hexo的next主题个性化教程:打造炫酷网站](https://www.cnblogs.com/php-linux/p/8416122.html)
[Hexo NexT主题中添加百度分享功能](https://asdfv1929.github.io/2018/05/25/baidu-share/)


---