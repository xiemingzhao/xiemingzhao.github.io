---
title: 使用Github+Hexo+Next免费搭建自己的博客（最细攻略）
categories:
- 博客搭建
tags:
- Hexo
- Next
copyright: true
abbrlink: GithubHexoNextblog
date: 2019-05-09
---

## 一、开篇
在这个信息爆发的时代，有着各种各样的社交平台和工具，而博客则是宣传和交流个人信息的一种重要方式。无论你是否从事IT行业，都可以通过博客来发布自己的一些学习所得、生活感悟或者喜怒哀乐。而目前主流的方式都是基于新浪微博、CSDN以及简书等，将人群割裂，且缺乏个性化和自主性。由其是对于大部分的互联网从业者，建立一个自己的博客网站是一件有意思的事情，当然偶尔也可以用来装个X。

刚刚转入互联网行业网的时候，一直就想拥有一个自己的博客，奈何一开始对这一类编程不是很熟悉，初次尝试失败后搁置了一些时间（现在才发现都是一些很简单的问题）。所以只要你想完成这个，抽点时间出来你肯定可以完成，又变成底子最好，没有也不要紧，照葫芦画瓢就行了。

话不多说先奉上本人的博客[小火箭的博客](www.xiemingzhao.com)。本人技术有限，但是以实用为主，本篇博文能够带领想我这样的小白使用Github+Hexo+Next一步一步的完成博客搭建。虽然目前搭建博客的方式有很多，但我们选择是目前最主流且体验下来感觉最舒适的一种。

**本文基于Windows 10 x64 专业版搭建，其他环境方法基本通用**

<!--more-->

## 二、git和Node.js的安装和配置
### 1. git安装
这一步比较简单，进入[git的官网下载页面](https://git-scm.com/downloads)，找到对应于自己的系统环境的安装包点击下载。

![gitweb](https://i.postimg.cc/Jzf6DRFZ/hexo1.jpg)

找到下载后的exe文件，点击就可以进入到安装页面了，如下图所示：

![gitexe](https://i.postimg.cc/DZFx6Bnz/hexo2.jpg)

剩下基本上是傻瓜式点点点，直到碰到下面这一个图：

![gitinstall](https://i.postimg.cc/SRxDFRXF/hexo3.jpg)

这一步是选择Git的操作路径环境，第一个就是只能在Git Bash中才能进行Git的操作，第二个则是将Git加入到系统环境中，这样你就可以在cmd中进行Git操作了。作为一个不断向大牛学习的小白，自然为了方便操作（毕竟cmd已经成了日常）会选择第二项。当然选择第一个也无碍，在需要git的地方右键调出`git bash`即可。

![gitbash](https://i.postimg.cc/rw0ZwDfq/hexo4.jpg)

### 2. Node.js安装
接下来安装`Node.js`一样的，进入[Node.js的官网下载页面](https://nodejs.org/en/download/)，找到对应自己的系统环境安装包点击下载。

![nodeweb](https://i.postimg.cc/503nRL2S/hexo5.jpg)

找到下载后的exe文件，点击就可以进入到安装页面了，如下图所示：

![nodeinstall](https://i.postimg.cc/d0GWGzMX/hexo6.jpg)

后面全部点点点，一直到最后完成即可。到这里，两个最重要的工具已经安装完成，我们打开控制台命令行，按Ctrl+R调出运行，输入cmd后回车即可。
![cmd](https://i.postimg.cc/wTBbnfJy/hexo7.jpg)

分别运行下列两行代码来检测上面两个工具安装是否正确：
```
git --version
node -v
npm -v
```
如果结果如下图一样显示各个工具的版本号，那么就说明安装完成。否则要回去检查哪一步安装错误。

![test](https://i.postimg.cc/tCmmb3tr/hexo8.jpg)


## 三、github账户的注册和配置
打开[github官网:https://github.com/](https://github.com/)，在如下图的注册页面中输入自己的注册信息，注意这里的用户名最好用一个标准化常用的，因为在后面创建代码库的时候需要保持统一。点击注册后，一定要去对应的邮箱中，将github官网发给你的确认邮件打开并确认注册才可以完成注册。

![githubweb](https://i.postimg.cc/DfxxQrD2/hexo9.jpg)

我们知道`github`是一个全球性的开发者项目开源交流平台，接下来就可以创建自己的代码库了。登录自己的账号后，点击下图中的+号，选择`New repository`进入代码库创建页面。然后你只需要在`Repository name`框中填入你的代码库名称即可。

**（格式为yourname.github.io，yourname是你上述注册的时候的用户名，也即页面上的Owner）**。例如，如果我的注册名是xiaoming，那么此处我就该填写`xiaoming.github.io`。后面的内容我们将以yourname来代替你的用户名，记得随时替换成自己的哦。

![githubcreate](https://i.postimg.cc/Ls0zYD57/hexo10.jpg)

在你点击`Create repository`正确创建代码库之后，你将会看到如下的页面：

![giothub.io](https://i.postimg.cc/nc5TgHpg/hexo11.jpg)

下面就需要打开`gh-pages`功能，这样才能进行后续的博客创建。我们点击页面功能区右侧的`Settings`选项，进入后下来找到`Github pages`模块，点击`Launch Automatic page generator`按钮，就会完成`gh-pages`页面的创建了。过几分钟后，尝试用浏览器访问一下`yourname.github.io`网址，如果可以正常打开页面，如下图所示（当然你的内容可能跟我不一样，因为这里已经是我添加了很多内容之后的了），那么Github部分的工作就完成了。

![ghpages](https://i.postimg.cc/TwLnrbWP/hexo14.jpg)

## 四、Hexo的安装和配置
本篇博客讲的是用Github+Hexo+Next来搭建博客，那么Hexo必不可少。这里是Hexo的官方文档[https://hexo.io/zh-cn/](https://hexo.io/zh-cn/)，有兴趣的可详细阅读。

### 1. Hexo的安装
首先在你的电脑本地选择一个盘，单独建立一个文件夹目录，方便用来管理后面的博客文件。假设你的目录是`E:/Blog`，那么这个时候将cmd的控制台操作目录cd到对应的目录下，可参考在控制台输入以下两行代码进行切换：
```
E：
cd Blog
```

这时候控制台的操作目录就变成了你的博客管理目录了。接着在命令行中分别运行下面的代码：
```
npm install hexo-cli -g
npm install hexo --save
```

**在第一行代码运行后也许会报出一个WARN，不用去管。**

最后在控制台输入下面的code来测试Hexo是否正常安装完毕：
```
hexo -v
```
如果显示出如下图的一系列工具版本号，那么就说明没什么问题了，Hexo安装到此结束。

![hexotest](https://i.postimg.cc/5NjnsHd8/hexo13.jpg)

### 2. Hexo的配置
接下来需要对Hexo进行初始化，建立一系列文件，在命令行中连续运行下面三行代码：
```
hexo init yourname.github.io
cd yourname.github.io
npm install
```

* 第一行是新建初始化目录，这一步名字也可以改，但是为了容易区分以及后续的多端管理博客建议可以这么命名。
* 第二行则是将工作目录换到初始化的目录下。
* 第三行是初始化环境，安装所需要的一系列文件。

运行完成后，对应目录下面应该多了很多文件，其中包含一下6个重要的文件或文件夹。

>node_modules
public          
source          
themes          
_config.yml 
package.json

### 3. Hexo初体验
到这一步Hexo基本上已经配置完成，可以进行初体验了。在控制台的命令行连续运行下列的代码：
```
hexo clean
hexo g
hexo s
```

运行结束你会得到一堆信息，其中包含一行关键信息如下：

>INFO Hexo is running at http://0.0.0.0:4000/. Press Ctrl+C to stop.

这就表明你的博客网站已经在本地开启了服务，这时候你可以尝试用浏览器访问网址[http://localhost:4000](http://localhost:4000)，如果能够正常访问你将会看到如下的初始博客页面。怎么样，颈不惊喜，意不意外，到这里你就完成了本地的基本框架的搭建和配置。

![localhost](https://i.postimg.cc/0ywZqLNQ/hexo12.jpg)

## 五、关联Hexo和Github Pages
利用Hexo来构建博客，必须要有对应的主机，并不是任何的电脑都能操作的。这时候就需要身份验证了，而这里使用的就是`SSH keys`。

### 1. 配置SSH keys
接上文，首先在命令行输入如下命令：
```
ssh-keygen -t rsa -C "youremail"
```
将代码中的youremail换位你注册github时候的邮箱。回车运行后会出提示让你输入一个密码，这个密码就是用来在以后体检项目的时候使用的，一版可以不用设置，省的自己忘了，毕竟已经限制了电脑。我们直接按回车，连续回车后运行结束。

下面你就可以去你电脑的C盘找到`C:\Users\bxm09\.ssh\id_rsa.pub`文件，这个文件里的内容就是刚才生成的秘钥，打开这个文件并原样复制里面的所有内容。再打开github的SSH keys配置页面[https://github.com/settings/ssh](https://github.com/settings/ssh)，点击 `new SSH keys`后，`Title`可以填写这个`SSH keys`的名字，例如mySHHkeys，然后将之前复制的内容原模原样的粘贴到`Key`里面，最后点击`Add SSH key`完成配置。

### 2. 测试SSH key
可以在命令行输入下列代码进行测试，**注意直接原样运行，git@github.com也不用更改**：
```
ssh -T git@github.com
```

你将会得到如下的信息：
```
The authenticity of host ‘github.com (207.97.227.239)’ can’t be established. 
RSA key fingerprint is 16:27:ac:a5:76:28:2d:36:63:1b:56:4d:eb:df:a6:48. 
Are you sure you want to continue connecting (yes/no)?
```

这时候直接输入`yes`就好了，你将会得到如下的信息，就说明配置成功：
```
Hi yourname! You've successfully authenticated, but GitHub does not provide shell access.
```

### 3. 配置个人信息
目前你的电脑可以连接到Github并进行操作了。Git会根据用户名和邮箱来记录你的每一次操作，所以需要对这些信息进行配置：
```
git config --global user.name "yourname"
git config --global user.email "youremail"
```
**记得将上述的`yourname`和`youremail`换位自己的用户名和邮箱哦**

### 4. 配置Deployment
上文中提到的6个主要文件，我们打开其中的`_config.yml`，找到`Deployment`配置的地方，按照如下的内容进行修改，改成你自己的信息，主要是`yourname`：
```
# Deployment## 标题 ##
## Docs: https://hexo.io/docs/deployment.html
deploy:
  type: git
  repo: git@github.com:yourname/yourname.github.io.git
  branch: master
```

### 5. 发文初体验
新建一篇博客，可运行下列的代码：
```
#paper-title是你的文章的标题
hexo new 'paper-title'
```

而后你就可以在你电脑的博客目录下的`/source/_posts/`文件夹中将看到刚才新建的`paper-title.md`文件。使用`markdown`编辑器编辑完成博客后就可以进行发布了。

使用Hexo发布文章提交到Github Pages的三条经典命令是：
```
# 删除旧的 public 文件
hexo clean

# 生成新的 public 文件
hexo generate

# 开始部署
hexo deploye
```

或者可以简写成如下也是等价的：
```
# 删除旧的 public 文件
hexo clean

# 生成新的 public 文件
hexo g

# 开始部署
hexo d
```

再或者可以将最后两条命令合并为一条：
```
# 删除旧的 public 文件
hexo clean

# 在部署前先生成
hexo d -g 
```

到这里你就完成了发布新博客文章的初体验，你可以访问你的博客地址`https://yourName.github.io`，将会展示你刚刚创建的文章。

**两个坑**

a. 如果在执行部署的命令时报错：

>deloyer not found:git

这时候需要提前安装一个扩展工具：
```
npm install hexo-deployer-git --save
```

b. 如果出现了如下的报错信息：

>Permission denied (publickey).
fatal: Could not read from remote repository.
Please make sure you have the correct access rights
and the repository exists.

这表明之前的`SSH key`没有配置好，重回到前面检查哪里不对，修复后再执行。

## 六、主题配置
Hexo可配置的主题特别多，本文选择了一个较为目前主流和简介的主题Next，本人的博客也是基于此主题构建的。该主题的作者也提供了详细的文档，感兴趣的可以多参考[Next官方文档](http://theme-next.iissnan.com/getting-started.html)。

### 1. 安装Next
Hexo安装主题比较简单，直接将主题温江拷贝到站点目录下的themes文件夹后，再进入站点配置文件中进行相应主题配置修改即可。这里以Next为例，我们依然将命令行路径切到你电脑的博客目录下，然后执行下列命令：
```
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

运行完成后进入对应的themes目录中即可发现next主题文件夹。这里介绍两个配置文件，分别是**站点配置文件**和**主题配置文件**，他们的名称都是`_config.yml`，但是内容和路径都不一样，前者是在博客站点根目录下面，后者是在`themes/next/`子目录的主题文件夹中。例如，我的两个配置文件目录分别是`E:/Bolg/myname.github.io/_config.yml`和`E:/Bolg/myname.github.io/themes/next/_config.yml`。前者是对Hexo本身的属性进行配置，后者则是对主题相关属性的配置。

### 2. 启动主题
Hexo启动任何主题的方法都一样，在完成克隆下载后，打开站点配置文件，找到themes配置字段，设置为对应的主题就可以了，例如这里配置如下：
```
# Extensions
## Plugins: https://hexo.io/plugins/
## Themes: https://hexo.io/themes/
theme: next
```
这时候就可以使用hexo发布提交的三条经典命令来测试效果了：
```
hexo clean
hexo g
hexo d
```

>这里提一个小技巧，因为在博客配置完善过程中，经常需要进行效果测试，每次都这么运行挺麻烦，一般都不使用`hexo d`而是使用`hexo s`在本地进行测试就行了，但是想要在`https://yourName.github.io`线上看到必须使用`hexo d`来发布啦。而且前两条命令也不是每次都用，不牵扯新文件生成的时候就可以不用`hexo g`，不需要清理public文件的时候就不要`hexo clean`，当然如果搞不清楚就都运行吧。

在使用`hexo s`后，与前文一致，打开本地的访问连接[http://localhost:4000](http://localhost:4000)就可以看到修改后效果了。到这里基本上完成了Github+Hexo+Next的博客基础框架的搭建。

### 3. 设置scheme
Next主题也有多种scheme，不同的scheme会呈现出不同的外观，并且主题其他的配置都可以使用不同的scheme。目前Next主要支持三种scheme:

>Muse - 默认 Scheme，横栏
Mist - 单栏外观
Pisces - 双栏设置

 在主题配置文件中搜索scheme，能够找到对应的配置位置，想要启动哪一个，就将其他的注释掉即可。例如博主使用的是Pisces风格，则设置如下：
```
#scheme: Muse
#scheme: Mist
scheme: Pisces
```

同样的，在这里也可以使用三个三个景点的发布语句，来测试不同scheme的效果，选择一个你最中意的即可。 

### 4. 设置站点
打开站点配置文件，在前面你就可以发现如下的`site`配置区域：

```
# Site
title: 我的博客
subtitle: 愿世界和平！！！
description: 小白的进阶
author: 小火箭
language: zh-Hans
timezone:
```
上面是我配置好的示例，每一项的配置内容都一目了然，对应的含义如下：
```
# Site
title: 你博客的大标题
subtitle: 你博客的副标题
description: 个人的描述
author: 作者名称
language: 语言类型
timezone: 时区
```
其他的没什么好说的，这里提一下language的配置。目前可用的语种有很多，这个示例中的`zh-Hans`是中文，详细可参考前面的Hexo官方文档。设置好站点的语言后，对应的一些关键词翻译可在主题文件中配置，打开`/themes/next/languages/`路径下的文件夹，可以找到对应的语言配置文件，例如中文的就是`zh-Hans.yml`。

到这里你可以仔细的看一下站点配置文件，内容并不多，而且大部分你都不用修改，例如`URL`模块是配置站点的链接地址生成格式的，`Directory`模块是配置站点的目录的，等等。

### 5. 设置菜单
设置菜单的时候，主要有三项：

>1. 菜单项的名称和连接配置
2. 对应主题语言的显示文本
3. 菜单项对应的图标

对于第一项，我们打开主题配置文件，找到`menu`的配置部分，如下所示，将你需要的模块取消注释，不需要的模块继续注释掉即可：
```
menu:
  home: / || home
  about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  archives: /archives/ || archive
  commonweal: /404/ || heartbeat
  others: /others/ || folder
```

上面是一个配置的示例，其中格式是item : link || icon，分别是菜单名称，连接以及图标。其中名称是不直接显示的，而是通过这个在语言配置中找到对应的翻译来显示。例如，你打开上述提到的主题配置文件中的语言配置文件，我这里的就是`zh-Hans.yml`，就可以找到对应主题各个部分的翻译，其中menu如下：
```
menu:
  home: 首页
  archives: 归档
  categories: 分类
  tags: 标签
  about: 关于
  commonweal: 公益404
  others: 其他
```

Next图标使用的是`Font Awesome`图标，其提供了上千种图标，基本上能够满足日常需求，详细图标可参考[Font Awesome官网](https://fontawesome.com/icons)。

上面的`others: /others/ || folder`是给的新增的示例，默认配置是没有的，可参考配置你想要的菜单模块，在主题配置中新增后，也需要在对应的语言文件中配置相关翻译。

这里需要提一下的时候，`menu`的配置还有另一种方式，你可以看到如下的`menu_icons`配置模块，是用来配置菜单图标的，如果你是如上述配置，图标已经在`menu`中通过`||`配置完成，这里只需要将enable设为true即可。
```
menu_icons:
  enable: true
```

或者你可以如下配置也是与上述等价的，只是将图标和内容分开配置了而已：
```
menu:
  home: /
  about: /about
  tags: /tags
  categories: /categories
  archives: /archives
  commonweal: /404
  others: /others
  
menu_icons:
  enable: true
  # Icon Mapping.
  home: home
  about: user
  categories: th
  tags: tags
  archives: archive
  commonweal: heartbeat
  others: folder
```

**切记配置的时候英文大小写要注意，非特殊说明一般都默认是小写**

### 6. 创建分类和标签等页面
虽然在上面我们配置了菜单中的categories和tags等等模块，但是对应的页面还是需要从新创建。方法比较简单，可以接着在上面的命令行中运行下面命令，也可以在本地的博客管理根目录中邮件选择`git bash here`调出git bash在其中运行下面的代码也是一样的效果，这就是之前我们安装的时候选择第二项的好处：
```
hexo new page categories
hexo new page tags
```
这时候会返回两条对应创建页面成功的信息，主要是`/source/categories/index.md`和`/source/tags/index.md`的文件路径。找到对应的文件路径，打开后，默认内容应该只有tile和date两个字段的配置，分别按照页面类型进行如下配置：
```
---
title: 文章分类
date: 2017-05-27 13:47:40
type: "categories"
---

---
title: 文章标签
date: 2017-05-27 13:47:40
type: "tags"
---
```
配置完成保存关闭即可。这时候在写文章的时候就可以加进去分类和标签了，并且发布后会再分类和标签页面进行展示，例如本篇博文的配置如下：
```
---
title: 使用Github+Hexo+Next免费搭建自己的博客（最细攻略）
date: 2019-05-09
categories:
- 博客搭建
tags:
- Hexo
- Next
---
```

**小技巧**,如果每次写文章都需要手动增加tags和categories两个单词很麻烦，我们打开`scaffolds/post.md`文件，在里面新增`tages:`和`categories:`两行就行了，这样每次执行`hexo new 'paper'`的后，文件里面就会自动包含这两个配置了。scaffolds目录下，是新建页面的模板，执行新建命令时，是根据这里的模板页来完成的，所以可以在这里根据你自己的需求添加一些默认值。好了，到这里你就学会和新增模块和对应的页面了。

### 7. 写作小配置
在hexo上发文章的是，一篇博文的大部分配置都在头部的配置区域中进行。上面提到了tile、date、tags等配置，以后会有更多的配置。

**多个标签**
一篇文章的分类和标签往往会有多个，多个标签怎么配置呢，如下所示两种方法等价：
```
tags: [a, b, c]
#或
tags:
  - a
  - b
  - c
```
分类的配置也是如此，但是不一样的是，多个tags的配置是评级的，多个categories的配置则是分级的，你可以亲自测试一下就了解了。

**阅读全文**
经常为了在首页更美观的展示博文的时候，我们对于每篇博文只会展示一部分信息，隐藏后面的，读者有兴趣了可以点击`阅读全文`再展开阅读。这个功能的实现也很方便，你只需要在想要折叠的部位插入一行`<!-- more -->`即可。

### 8. 侧栏
侧栏默认是在有目录列表的情况下才会显示，如果你如上述配置，这时候使用那三条测试命令一定能够访问到你的页面，并且展示出侧栏的。侧栏的配置主要有两个属性，如下所示，找到`sidebar`模块就是配置侧栏的位置：
```
sidebar:
  # Sidebar Position, available value: left | right (only for Pisces | Gemini).
  position: left
  #position: right

  # Sidebar Display, available value (only for Muse | Mist):
  #  - post    expand on posts automatically. Default.
  #  - always  expand for all pages automatically
  #  - hide    expand only when click on the sidebar toggle icon.
  #  - remove  Totally remove sidebar including sidebar toggle.
  display: post
  #display: always
  #display: hide
  #display: remove
```
其中配置内容和注释都特别详细，一个属性是`position`，侧栏的位置，两个选择放左边还是右边，按照你的个人喜好来选择即可。另一个属性则是`display`，侧栏的展示模式，有四种配置，翻译一下就是：
```
post - 默认，在文章页面（拥有目录列表）时显示
always - 总是显示
hide - 总是隐藏（可以手动展开）
remove - 移除
```

### 9. 设置头像
在这个时代，一个博客怎么能没有彰显自己个性的头像呢。Next主题设置头像也是很简单的，依然在主题配置文件里面，搜索`avatar`字段，找到头像配置模块，如果搜索不到，可能是版本太老，可以自己新添加。如下配置示例：
```
# Sidebar Avatar
# in theme directory(source/images): /images/avatar.gif
# in site  directory(source/uploads): /uploads/avatar.gif
avatar: https://example.com/youravatar.jpg
```
上述的`https://example.com/youravatar.jpg`就是对应头像的连接，换成你自己的就可以了。这里有两个方式，一个是如此例一样，已连接形式配置，如果是本地图片可上传到第三方图床上生成连接。另一种方法就是存在站点内地址，将头像放置主题目录下的 `source/uploads/` （新建uploads目录若不存在） 配置为：`avatar: /uploads/avatar.png` 或者 放置在 `source/images/` 目录下 , 配置为：`avatar: /images/avatar.png`。

### 10. 添加插件
这里添加的插件是指sitemap和feed插件，具体有何用，大概就是可以生成博客的RSS，在连入互联网被一些搜索网站收录后，能够快速的建立站点底图，将你的页面展示在搜索结果中。

话不多说，我们开始，这里需要将控制台切到你博客的根目录下面，执行下面两条命令：
```
npm install hexo-generator-feed -save
npm install hexo-generator-sitemap -save
```
在这之后需要打开站点配置文件`_config.yml`，新增以下内容，当然提前先搜索以下有没有这个模块，如果有的话就直接修改就好了，目前的版本是没有的，然后随便找个空白地方粘贴进去即可：
```
# Extensions
Plugins:
- hexo-generator-feed
- hexo-generator-sitemap
#Feed Atom
feed:
  type: atom
  path: atom.xml
  limit: 20
#sitemap
sitemap:
  path: sitemap.xml
```
最后执行三条境地啊匿名了来发布测试：
```
hexo clean
hexo g
hexo d
```

都完成之后，你就会在自己的github主目录下[https://gdutxiaoxu.github.io](https://gdutxiaoxu.github.io)找到两个新增的文件，`atom.xml`和`sitemap.xml`，并且博客侧栏头像下面应该会多出一个`RSS`模块。

### 附加： 404页面
每个网站都应该有自己的404页面，404大家都不陌生也就是Not Found的意思。为自己的博客添加404页面也是比较简单的。直接在博客的文件主目录（一般就是根目录的source文件夹`/source/`）下面新增一个`404.html`文件，然后在里面构建自己的页面内容即可。
**需要注意的是404页面仅仅对绑定顶级域名的项目才起作用，像github这种默认分配的二级域名是不起作用的，及时你用`hexo s`在本地测试也不行**

大部分博主都喜欢使用[腾讯公益404](https://www.qq.com/404/)的页面作为自己的404页面，这也不为一个不错的idea，这里我们也以此为例吧，可以在`404.html`文件中新增：
```
<script type="text/javascript"
        src="//qzonestyle.gtimg.cn/qzone/hybrid/app/404/search_children.js"
        charset="utf-8"
        homePageUrl="http://www.lovebxm.com/"
        homePageName="回到我的主页">
</script>
```

### END
其实建站的方案有很多，例如：
>[Hexo + GitHub Pages ](https://hexo.io/zh-cn/)
[Jekyll + GitHub Pages ](https://www.jekyll.com.cn/)
[WordPress + 服务器 + 域名](https://cn.wordpress.org/)
[DeDeCMS + 服务器 + 域名 ](http://www.dedecms.com/)
...

但是个人最喜欢Hexo + GitHub方案，一来是免费，二来是能够和自己的代码库一起管理。但是其也有自己的缺点，那就是不支持数据库管理，所以你只能做静态页面的博客，不能像其他博客（如 WordPress）那样通过数据库管理自己的博客内容。 但是GitHub Pages 无需购置服务器，免服务器费的同时还能做负载均衡，github pages有300M免费空间。静态的博客更有利于搜索引擎蜘蛛爬取，轻量化的感觉真的很好。 

**参考博文**
[手把手教你用Hexo+Github 搭建属于自己的博客-gdutxiaoxu](https://blog.csdn.net/gdutxiaoxu/article/details/53576018)
[可能是最详细的 Hexo + GitHub Pages 搭建博客的教程-QQ80583600](https://blog.csdn.net/qq80583600/article/details/72828063)