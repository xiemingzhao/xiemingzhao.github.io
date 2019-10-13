---
title: Hexo博客多设备管理
categories:
- 博客搭建
tags:
- Hexo
copyright: true
abbrlink: Hexoblogbranch
date: 2019-05-27
---

想要小白详细版本的Github+Hexo+Next搭建博客教程，可访问我的另一篇博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/GithubHexoNextblog)。

## 多设备管理博客
博客建立好之后，面临的就是维护和更新博客，但是总不能每次都带着自己的电脑吧，如果想在自己的办公电脑上也操作自己的博客呢？于是广大的IT人才们想出了构建github分支来管理自己的博客。

### 1. github 创建分支
不管你有无看过我的另一篇博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/GithubHexoNextblog)，相信你自己博客的时候都是在yourname.github.io仓库下的master分支上创建的。既然我们是在本地上管理离线文档的，那我们就领创建一个分值，只用来存储每次更新的离线文件，相当于一个本地了。

如下图，在分支的输入框中新建一个分支确认就好了，我这里已经新建好了，一般取名hexo打扰其他的也都OK：

![branch1](https://i.postimg.cc/yxKQvdvC/branch1.jpg)

然后我们需要进入`settings`模块，如下图，再点击`Branches`，接着选择默认分支为你刚才新创建的，我这里就是`hexo`，毕竟以后体检离线文件是要提交到这里的，所以将它设为默认分支，保存就好了。

![branch2](https://i.postimg.cc/3JKfGkkg/branch2.jpg)

### 2. 本地clone分支
注意哦，到目前还未使用新电脑，既然要使用新电脑进行备份更新，那我们先在本地也完成备份更新的步骤，后面新电脑操作跟本地差不多。我们在你博客管理目录中新建一个分支管理目录，例如，我的目录下面就有一个创建的目录`myname.github.io`以及一个分支目录`hexo`，之后基本上就只使用`hexo`分支，换电脑时候也类似只会有一个`hexo`分支来更新管理。

创建好新的目录后，使用控制台进入博客根目录（也就是能够看到主分支和hexo分支的目录）执行下面的命令，将分之内容clone到本地分支：
```
git clone -b hexo https://github.com/yourname/yourname.github.io
```

然后本地的hexo分支文件夹中会多了很多文件，进入hexo文件夹中右键掉出`git bash`执行`git branch`，会返回你当前所处的是`hexo`分支。

这里去站点配置文件`_config.yml`中检查一下branch的配置是否为主分支：
```
deploy:
    type: git
    repo: https://github.com/yourname/yourname.github.io
    branch: master
```

### 3. 备份到分支
进阶上述完成clone分支后，我们需要删除一些文件，因为我们需要的是备份部署资源，其他在发布生成过程中产生的静态文件并不需要保留。首先我们删除除了**.git**文件夹的其它所有文件和文件夹，主要是为了得到版本管理的.git。当然，最新版好像这一部分也不需要保留了，那就先保留着吧，到时候部署的时候不行的话再删也不迟。

接着把本地主管理目录（也就是`yourname.github.io`的文件夹）一下7个文件/文件夹复制到本地分支目录`hexo`中去：
```
scaffolds/ #文件夹（文章的模板）；
source/ #资源文件夹；
themes/ #文件夹里面的主题；
.git/ #版本管理
.gitignore #限定在提交的时候哪些文件可以忽略
_config.yml #站点配置
package.json #说明使用哪些包；
```

如下图所示，分支文件夹中应该包含上述的文件，我这里有一些其他的是因为已经使用了很久，添加了其他功能，部署的时候产生的：

![branch3](https://i.postimg.cc/NfpJkQXZ/branch3.jpg)

不需要更新的文件可利用.gitignore文件来配置需要忽略备份的文件：
```
/.deploy_git/  
/blogSource/  
/node_modules/  
/public/  
*.db  
*.json  
```
*不过最新版一版不用配置，文件中应该已经有了*

这里可执行一下：
```
npm install
```
防止后续的部署会失败，而所依赖的包都在上一步中的package.json备份文件里，所以直接这一个命令就可以了。

**注：**如果使用的主题是从GitHub克隆的，那么主题文件夹下有Git管理文件，需要将它们移除，我使用的是hexo-next，需要移除的文件是`themes/next/.git*`。

**不要hexo init去整体初始化，因为需要的文件我们已经拷贝过来了。**

### 4. 提交分支和更新
我们使用控制台进入hexo分支目录或者在hexo分支目录里面调出`git bash`一次运行以下命令：
```
git add .  #后面的点要加哦
git commit -m ‘新电脑部署’  #引号内的内容是对每次提交的备注
git push  #推送文件,保证xxx分支版本为最新版本
```
以后每次更新，本地可使用`hexo`分支来操作，每次操作完将修改的部署文件提交到备份分支就可以了，具体可以使用如下命令：
```
git pull  # 保持本地操作为最新版本
hexo n xxx #编辑、撰写文章或其他博客更新改动
hexo clean # 可选，如果配置文件没有更改，忽略该命令
hexo g
hexo s #本地测试
hexo d
git add .
git commit -m ‘在新电脑上提交新文章’
git push #保证xxx分支版本为最新版本
```
**如果上述中间出错，可删除`.git/ #版本管理`文件后重试。**

### 5. 新电脑迁移
前面讲了一大堆已经实现了分制管理，且每次本地主机也是拉取线上最新的，更改部署后，也将最新的版本提交到线上。这样使得真个流程先关起来，每次都是对最新版本进行更新，线上`hexo`分支目录存储的也都是最新版本。

如果现在你项实用另一台电脑操作你的博客，首先你需要做的是底层工具安装：

>安装 Git
安装 Node.js

这里不清楚的可以参考我之前的博客[使用Github+Hexo+Next免费搭建自己的博客（最细攻略）](https://www.xiemingzhao.com/posts/Github+Hexo+Next_blog)。

然后需要将新电脑生成`SSH key`配置到github上，这样新电脑才能得到信任，与线上github进行相连。这一步不会的也可以参考上述提到的我的另一篇博客。

然后在新电脑本地创建一个管理目录，命名为hexo也行。紧接着将线上的`hexo`分支clone到本地，这一步与上述一样。然后安装`hexo`相关依赖：
```
npm install
```
如此便完成了将Hexo博客迁移到新电脑上了，很简单。后面更新部署和发表新文章就可以使用上一步中那几条命令。**切记每次要拉取和更新线上最新版本哦。**

**参考博文**
[GitHub创建Git分支将Hexo博客迁移到其它电脑](https://blog.csdn.net/white_idiot/article/details/80685990)
[hexo 迁移更换电脑，或多电脑终端更新博客](https://andyvj.coding.me/2019/02/19/190219-03/)




