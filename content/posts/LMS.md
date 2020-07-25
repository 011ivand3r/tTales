---
title: "LMS"
date: 2020-07-25T13:39:03+05:30
cover:
    image: /tTales/imgs/dice.jpg
    style: wide
mathjax: true
tags:
  - Machine Learning
  - Probability
  - Regression
draft: false
---
The first thing whenever one starts learning about Machine Learning algorithms is Linear Regression. But the way people generally blaze through it because it's *rudimentary* and label it as *very intuitive* doesn't sit well with me.

So, in this article we will build a Linear Regression model with the help of probability.

**Regression's** idea is to determine the strength of relation between a bunch of independent variables $x = (x_1, x_2, ... x_n)$ and a dependent variable $y$ (i.e. target variable).

So, generally the Linear Regression hypothesis is represented as --
$$ h_{\theta} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n $$
$$ \therefore h_{\theta} = \sum_{i=1}^{n} \theta_ix_i $$
And in vector form we can write the same as --
$$ y = h(x) = \theta^Tx  $$
Note: $h_\theta(x)$ is same as $h(x)$

In a Machine Learning problem, we can have different kinds of errors. For example, we can have errors if we don't consider some feature(s) to determine the result, although it is very relevant to the problem. It can also be a measurement error, or a random noise. So, the target variable can be written as --
$$ y^{(i)} = \theta^Tx^{(i)} + \epsilon^{(i)} $$

$\epsilon^{(i)}$ accounting for the errors in our model. We assume that the error term of each training example, $\epsilon^{(i)}$ is Independently and Identically Distributed (IID). And they satisfy a Normal Distribution with mean 0, and standard deviation $\sigma$.

And now you might ask *why Normal(Gaussian) Distribution? Why not anything else?*

The answer is given by [**Central Limit Theorem**](https://www.youtube.com/watch?v=YAlJCEDH2uY). Which tells us that if you take sufficiently large number of random numbers, then the sum of the numbers will be approximately normally distributed. So, this means that the error of one training example has the same distribution as the error of other training examples.

Also, $y^{(i)}$ is a random variable that satisfies Normal distribution with the mean $\theta^Tx$ and variance being $\sigma^2$. Now if we write the probability distribution of the predicted values $y$, given $X$ and the parameter $\theta$. It is given as follows:
$$p(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt2\pi\sigma}exp(-\frac{(y^{(i)}- \theta^Tx^{(i)})^2}{2\sigma^2})$$
This means that the distribution of $y^{(i)}$ given $x^{(i)}$ is parameterized by $\theta$.

Now here is where, we get introduced directly to the **Ordinary Least Squares**, and we try to find the most likely $\theta$ for a given set of $(y^i, x^i)$ pairs by reducing the cost function. But, a more insightful way to find this estimate is to maximize the [**likelihood function**](https://en.wikipedia.org/wiki/Likelihood_function).

*So, what is **Likelihood function?***

It is kind of a negative of loss function. But, fundamentally speaking it is the probability distribution function of $\theta$.

![](/tTales/imgs/Likelihood.png)

The picture above taken from [Statquest wit Josh Stammer's](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw) YouTube channel give the easiest idea of what Likelihood is about. You can check out his [video](https://www.youtube.com/watch?v=pYxNSUDSFH4) for more details.

 Since we assume that every training example is IID, we can write for n training examples --

 $$L(\theta) = \prod_{i=1}^{n} p(y^{(i)}|x^{(i)}; \theta)$$
 $$\implies L(\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt2\pi\sigma}exp(-\frac{(y^{(i)}- \theta^Tx^{(i)})^2}{2\sigma^2})$$

We need to maximize this function in order to find the parameter values that give the distribution that maximizes the probability of observing the data. The parameter values(i.e. $\theta$) are found such that they maximize the likelihood that the process described by the model produce the data that were actually observed.

To find this function's maximum is quite tough, so we can find the maximum of the $\log$ of the Likelihood function. This is only possible because the Likelihood function and the log of the Likelihood function both peak at the same point for the same $\mu$ and $\sigma$. Also, it's way, way easier to find the derivative.

So, let's find the derivative.

\begin{align*}
$$ L(\mu|y)} &= afdsaf \\
&= btcvgrw\\
&= sfsfasdff$$
\end{align*}

As, the first term is a constant anyway so to increase the function, is to maximize the second term. The term which we may already know as the cost function. Written as --
$$J(\theta) = \frac {1}{2} \sum_{i=1}^{n} (y^{(i)}- \theta^Tx^{(i)})^2$$

Now, it feels like cost function was always meant to be the cost function. It didn't just drop from anywhere. And suddenly you find yourself asking your intuition to agree that half of the sum of the squares of the distances of prediction from the actual value, is need to be reduced to get a better parameter, i.e. $\theta$.

If you have come this far, please leave a comment and a reaction. Let me know, if I made any mistake. Thanks!
