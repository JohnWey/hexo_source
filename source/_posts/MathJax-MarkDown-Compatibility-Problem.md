---
title: MathJax 和 MarkDown兼容性的问题
---

由于下划线(\_)以及反斜线(\\) 是MarkDown的特殊字符，而这些字符在MathJax分别代表不同的含义（前者表示下标，后者双反斜线表示换行\\\\）,因此在渲染数学公式时，MarkDown会将这些字符作为其特殊字符，导致显示公式凌乱。

网上很多解决方案已过时，为此折腾了好几天，很没必要，毕竟应该多花心思在写blog上，而不是如何配置它。这里提供另外一套解决方法：
1. 放弃next主题，使用hexo-theme-paperbox代替[github](https://github.com/sun11/hexo-theme-paperbox)
2. 修改该主题下\_config.yml文件，配置mathjax: true
3. 卸载默认的mark渲染组件，安装新的渲染组件
   npm uninstall hexo-renderer-marked --save 
   npm install hexo-renderer-kramed --save
4. 更新站点并重启服务
   hexo clean
   hexo g -d
   hexo s

需要注意的是：对于需要在某一行显示数学公式的情况，只加$...$这样是不行的，需要在两个$符号首尾加上（\`）号，如：\`$....$\`