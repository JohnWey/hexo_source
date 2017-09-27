---
title: 机器学习第一周总结——Linear Regression
---

先搞清楚什么是机器学习，这对定位问题是否应该使用机器学习来解决很重要，有些问题完全没必要使用机器学习，就没必要杀鸡用牛刀了。

----------


###什么是机器学习?
- *Arthur Samuel(1959):*
> Field of study that gives computers the ability to learn without being explicitly programmed.
> **说人话：不需要太多的编程就能使计算机拥有学习某一领域的能力。**  

- *Tom Mitchell(1998):*
> A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.
> **说人话：AlphaGo为了完成人机对战（下棋 T），不断学习下棋，从中获取经验（E），目的是为了提高和人比赛的胜率（P）。**

----------


**机器学习分类**
机器学习涉及的范围很广，针对不同的问题，学习策略也很多，但总体而言大致可以分为**监督学习**和**非监督学习**两种（还有半监督学习和强化学习）。不同的问题需要使用不同的策略。

还是把这两种策略的含义搞清楚先：
*监督学习？*
> 可以理解为**一部分数据的答案已经知道了**。比如我们要预测未来大盘的点位，在历史中大盘的点位已经知道了；再比如我们要让机器知道给它的图片是个帅哥还是美女，前提是我们已经知道了这个图片是帅哥还是美女。

可想而知，如果需要人工去分类（打标），是一个多大的工程。这也孕育出了很多以出售打标数据盈利的公司。

##### **什么是无监督学习？**
> 和监督学习相反，**数据的答案事先我们不知道**。而寻找答案是一个无中生有的过程。比如你遇到一群外星人，这群外星人各自有着不同的特征，你需要通过聚类的方法把有相同特征的外星人分组到一起，然后研究他们哪些对人类友好，哪些对人类有威胁，这叫无监督学习。

*算法一览：*

| 机器学习种类 | 算法分类   | 算法                                       |
| :----- | :----- | :--------------------------------------- |
| 监督学习   | 分类、 回归 | K近邻、朴素贝叶斯、决策树、随机森林、GBDT和支持向量机；     线性回归、逻辑回归 |
| 无监督学习  | 聚类、 推荐 | K-Means、DBSCAN、协同过滤                      |
| 半监督学习  | 聚类、 推荐 | 标签传播                                     |
| 强化学习   |        | 隐马尔可夫                                    |

----------
#### **1. 能预测未来的神奇算法——线性回归**

说到线性回归，其实我们初中的时候就学过它的简单方程式，只不过那会儿我们没有安利这样一个高大上的名字，我们那会儿叫**斜截公式**：
$$
y = kx + b
$$
来看看只有一个参数的**单参数线性回归模型**：
$$
h_\theta(x^i)=\theta_0+\theta_1 x_1^i  (其中 x_0^i=1，x_1^i表示一个特征，这个特征有i个值 )
$$
所谓的特征就是二维表中具有计算意义的一列数据，比如

| ID   | Sex  | High   |
| :--- | :--- | :----- |
| 1    | 男    | 170 cm |
| 2    | 女    | 175 cm |
| 3    | 男    | 180 cm |
| 4    | 女    | 200 cm |
其中Sex是一个特征，High是另一个特征，ID不是个特征，它只是个序列索引而已。他们的i都是4，因为有4条数据。是不是秒懂？：） 是的，在一元线性回归算法中，我们就是用一条倾斜的直线来**预测未来**。原来我们从初中时就可以预测未来了。：）

##### **为什么我们可以用类似一条直线来预测呢？**
这个问题也可以换个说法，**在什么情况下可以使用线性回归算法？**
> 1. 看数据的分布是有一定规律的，可以通过直线或曲线来拟合数据的中心。
> 2. 需要预测的变量是连续的值，比如房价，股票价格。而不是离散值，比如只有男、女等。

再来看看多参数的**线性回归模型**：
$$
h_\theta(x^i)=\theta_0 x_0^i+\theta_1 x_1^i+\theta_2 x_2^i \cdots+\theta_n x_n^i
$$
用**向量表示**：
$$
h_\theta(x)=\theta_0\begin{bmatrix} x_0^1 \\x_0^2\\ \vdots \\x_0^n \end{bmatrix}
+\theta_1\begin{bmatrix} x_1^1 \\x_1^2\\ \vdots \\x_1^n \end{bmatrix}
+\cdots
+\theta_1\begin{bmatrix} x_1^1 \\x_1^2\\ \vdots \\x_1^n \end{bmatrix}
$$
用**矩阵**表示：
$$
H=\begin{bmatrix} 
x_0^1 & x_1^1 & x_2^1  & \cdots & x_n^1 \\
x_0^2 & x_1^2 & x_2^2  & \cdots & x_n^2 \\
x_0^3 & x_1^3 & x_2^3  & \cdots & x_n^3 \\
\vdots & \vdots & \vdots & \ddots & \vdots\\
x_0^n & x_1^n & x_2^n  & \cdots & x_n^n \\
\end{bmatrix}* 
\begin{bmatrix}\theta_0 \\ 
\theta_1 \\ 
\theta_2 \\ 
\vdots \\
\theta_n \\
\end{bmatrix}=X\theta
$$
简不简单？有了这个公式，我们就能**预测未来**了：）

----------
#### **2. 如何预测？**
上面那个模型中`$x$`是确定的，即我们的各种特征数据，不确定的是`$\theta$`值。只要找到了`$\theta$`，我们就可以写出那个模型方程式，再把新的数据代入到$x$中，就知道了**未来**。所以，如何算出`$\theta$`?

针对一元回归模型，不同的`$\theta$`意味这不同的斜率，不同的斜率他们和真实数据的拟合程度是不一样的。如何确定`$\theta$`使得预测的误差最小呢？
$$
J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h\theta(x^i)-y^i)^2
$$

其中`$y^i​$`表示真实数据，而`$h_\theta(x^i)​$`表示预测数据。
**说人话：`$\theta$`要满足这样的条件，即预测出来各个点的值与真实值之间的差的平方和最小。**

现在的问题就转换成了**求`$J(\theta)$`的最小值**问题了！
即：
$$
\min_{\theta_1,\theta_2 \cdots\theta_n}J(\theta_1,\theta_2,\cdots,\theta_n)
$$
这个问题有两种解决方案：

##### **1. 梯度下降**
要了解梯度下降算法，首先要知道求导公式的意义：
$$
 \frac{\partial J}{\partial \theta}=\frac{\Delta y}{\Delta x}=tg \alpha
$$
**说人话：每变化一点点`$\theta$`，随之而变的`$J$`变化了多少？**
还是没懂！？ 上一个百度图片：

>![这里写图片描述](http://img.blog.csdn.net/20170914173743114?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSm9obldleQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
>图片中`$\Delta x$`就是变化的那一点点（对应`$\partial \theta$`或`$\Delta y$`），而对应的曲线`$f(x)$`，发生了`$\Delta y$`这么多变化。当N和M非常接近时（即`$\Delta x$`很小很小），我们可以用PQ的高度（`$dy$`）来近似NM的高度（其高度`$dy$`=在`$x_0$`处的斜率 * `$\Delta x$`）。

知道了这个我们再来看看**梯度下降算法公式**：
$$
\theta_j := \theta_j - \alpha\frac{\partial J}{\partial\theta_j}
$$
公式里的`$\frac{\partial J}{\partial\theta_j}$` 可以简单的理解为上图中的tga，`$\alpha$`是个正常数，学名叫**学习速率**。这个tga很神奇，在小于90°，它是个正实数；在大于90°时是个负实数。

所以，对于梯度下降算法公式，
当T倾斜向上，即角度小于90°时，`$\alpha \frac{\partial J}{\partial\theta_j}$`值为正，`$\theta_j$`从大变小，`$\frac{\partial J}{\partial\theta_j}$`不断的趋近于0，`$\theta_j$`不断的向左移动减小，直到移动到曲线的底部；
![这里写图片描述](http://img.blog.csdn.net/20170914183926463?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSm9obldleQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
当T倾斜向下时，即角度大于90°时，`$\alpha  \frac{\partial J}{\partial\theta_j}$`值为负，`$\theta_j$`从小变大，`$\frac{\partial J}{\partial\theta_j}$`不断的趋近于0，`$\theta_j$`不断的向右移动增大，直到移动到曲线的底部
![这里写图片描述](http://img.blog.csdn.net/20170914184358503?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSm9obldleQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

理解了原理，下面就是如何算`$\frac{\partial J}{\partial\theta_j}$`了：
$$
\frac{\partial J}{\partial\theta_j}=\partial\frac{\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2 }{\partial\theta_j}
$$

把`$\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2$`展开：
$$
J=\frac{1}{2m}[(h_\theta(x^1)-y^1)^2+h_\theta(x^2)-y^2)^2+\cdots+h_\theta(x^m)-y^m)^2]
$$
**注意：上面的`$x^1$`表示第一个训练样本，而`$y^1$`是第一个训练样本所对应的真实目标值。**
我们对其中一个子式子继续展开研究：
$$\frac{\partial(h_\theta(x^1)-y^1)^2}{\partial\theta_j}=\frac{\partial[(\theta_0x_0^1+\theta_1x_1^1+\cdots+\theta_nx_n^1)-y^1]^2}{\partial\theta_j}$$
$$
=2((\theta_0x_0^1+\theta_1x_1^1+\cdots+\theta_nx_n^1)-y^1)x_j^1
$$


简化成向量的形式：
$$
\frac{\partial(h_\theta(x^1)-y^1)^2}{\partial\theta_j}=2( \begin{bmatrix} x_0^1&x_1^1&\cdots&x_n^1\end{bmatrix} \begin{bmatrix} \theta_0\\ \theta_1 \\ \vdots \\ \theta_n\end{bmatrix} - y^1)x_j^1==2(X^1\theta-y^1)  x_j^1
$$
所以最终我们的梯度下降`$\theta​$`参数的确认式子为:
$$
\theta_j := \theta_j - \alpha \frac{1}{2m} \sum_{i=1}^m(2(X^i\theta-y^i)x_j^i) 
=\theta_j- \alpha\frac{1}{m}\sum_{i=1}^m((X^i\theta-y^i)x_j^i)
$$
$$
(j=0,1,...,n)
$$

对求和公式展开后写成矩阵的形式：
$$
\theta:=\theta-\alpha \frac{1}{m}((X^1\theta - y^1)
\begin{bmatrix} 
x_0^1 \\
x_1^1 \\
\vdots \\
x_n^1
\end{bmatrix}
+(X^2\theta - y^2)
\begin{bmatrix} 
x_0^2 \\
x_1^2 \\
\vdots \\
x_n^2
\end{bmatrix}
+\cdots\\
+(X^n\theta - y^n)
\begin{bmatrix} 
x_0^n \\
x_1^n \\
\vdots \\
x_n^n
\end{bmatrix})  
(其中\theta为列向量)
$$

$$
\theta:=\theta-\alpha \frac{1}{m}
\begin{bmatrix} 
x_0^1 & x_0^2 & \cdots & x_0^n\\
x_1^1 & x_1^2 & \cdots & x_1^n\\
\vdots & \vdots & \ddots & \vdots \\
x_n^1 & x_n^2 &  \cdots & x_n^n
\end{bmatrix}
\begin{bmatrix}
X^1\theta-y^1\\
X^2\theta-y^2\\
\vdots \\
X^n\theta-y^n
 \end{bmatrix}
$$
**终极公式：**
>$$\theta:=\theta-\alpha \frac{1}{m}(X^T(X\theta-Y)) \tag{*}$$
>什么意思？意思是是只要不断的迭代`$\theta$`，最终`$\alpha\frac{1}{m}(X^T(X\theta-Y))$`会收敛到0，从而得到一个收敛后的`$\theta$`，获得最小值。

##### **2. 正规方程**
正规方程可以更快速简单的求解`$\theta$`值，再一起推导一下。
首先正规方程也是从这个方程而来：
$$ J(\theta)=\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i)-y^i)^2=\frac{1}{2m}[(h_\theta(x^1)-y^1)^2+\cdots+h_\theta(x^m)-y^m)^2]$$
因为
`$X^TX=X^2$`
所以
$$
J(\theta)=\frac{1}{2m}[(h_\theta(x^1)-y^1)^T(h_\theta(x^1)-y^1)+\cdots+(h_\theta(x^n)-y^n)^T(h_\theta(x^n)-y^n)]
$$

$$
=\frac{1}{2m}
\begin{bmatrix} 
(h_\theta(x^1)-y^1)^T \\
(h_\theta(x^2)-y^2)^T \\
\vdots \\
(h_\theta(x^n)-y^n)^T \end{bmatrix}
\begin{bmatrix} 
(h_\theta(x^1)-y^1) &
(h_\theta(x^2)-y^2) &
\cdots &
(h_\theta(x^n)-y^n)
\end{bmatrix}
$$

$$
=\frac{1}{2m}(X\theta-y)^T(X\theta-y)
$$

$$
=\frac{1}{2m}[\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^Ty]
$$





对`$\theta$`求导，且求导后的值要趋于0，所以：
$$
\frac{\partial J}{\partial \theta}=\frac{\partial(\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^2)}{\partial\theta}=0
$$


因为`$\theta^TX^Ty=y^TX\theta$`所以有：
$$
\frac{\partial(\theta^TX^TX\theta-\theta^TX^Ty-y^TX\theta+y^2)}{\partial\theta}=X^TX\theta-2X^Ty=0
$$

$$
(X^TX)^{-1}(X^TX)\theta=(X^TX)^{-1}X^Ty
$$
$$
\theta=(X^TX)^{-1}X^Ty \tag{*}
$$

**梯度方法和正规方程方法比较：**

| 梯度下降方法 | 正规方程   | 
| :------------------| :------------------------------- | 
| 适合特征大于1W的情况   |  适合特征小于1W的情况  |
| 需要归一化（特征标准化）|  不需要归一化   | 
| 方法相对复杂          |  方法简单      |



##### ** 注：以上数学推导过程若有不严谨之处，欢迎指出！**

