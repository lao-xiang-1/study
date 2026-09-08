
## 1. MLE

**core idea**: find the parameter $\theta$ that maximizes the likelihood of the observed data

>Given the known form of the distribution (e.g., normal distribution, binomical distribution), infer the most probable values of the distribution parameters from the observed samples.

### 1.1 Likelihood Function

Let $x_1, x_2, \dots, x_n$ be the independent and identically distributed samples with parameters $\theta$, then the  likelihood function is defined as the joint probability of the samples:
$$
L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta)
$$
MLE seeks the parameter $\theta$ that maximizes $L(\theta)$:
$$
\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta)
$$

## 2. Log-Likelihood Estimation

### 2.1 why log?

it's very complicated to differentiate the product form of $L(\theta)$ directly. Taking the logrithm converts **the product to sum**, greatly simplify the computation:
$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^{n} \ln P(x_i \mid \theta)
$$
The optimization objective becomes:
$$
\hat{\theta} = \arg\max_{\theta} \ell(\theta)
$$

> Clearly $\hat\theta$ remains unchanged, since $ln(x)$ is a monotonically increasing function.


## 3. Example: Parameter Estimation for the Normal Distribution

Let $X \sim \mathcal{N}(\mu, \sigma^2)$, with samples $x_1, x_2, \dots, x_n$

### 3.1 Likelihood Function
$$
L(\mu, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

### 3.2 Log Likelihood Function
$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - n\ln\sigma - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2
$$


### 3.3 Solving by Differentiation

Take the partial derivative with respect to $\mu$ and set it equal to 0:
$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^{n}(x_i - \mu) = 0
$$

Solving gives:
$$
\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

Take the partial derivative with respect to $\sigma^2$ and set it equal to 0, gives:
$$
\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^{n}(x_i - \hat{\mu})^2
$$

> Note: Here $\hat\sigma^2$ is a biased estimation, while the sample variance $S^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ is fthe unbiased estimaiton.


## 4. Example: Parameter Estimation for the Poisson Distribution

Suppose the number of calls received per hour at a call center follows a Poisson Distribution $X \sim \text{Poisson}(\lambda)$. Over 5 hours, the observed call counts are: $3, 5, 4, 2, 6$. Use MLE to estimate the parameter $\lambda$.

The probability mass function (PMF) of Poisson Distribution is:
$$
P(X = k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots
$$

### 4.1 Likelihood Function

Since the samples $x_1, x_2, \dots, x_n$ are independent, their joint probability is:
$$
L(\lambda) = \prod_{i=1}^{n} \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} \cdot e^{-n\lambda}}{\prod_{i=1}^{n} x_i!}
$$

### 4.2 Log Likelihood Function

Taking logrithm gives:
$$
\ell(\lambda) = \ln L(\lambda) = \left(\sum_{i=1}^{n} x_i\right) \ln\lambda - n\lambda - \sum_{i=1}^{n} \ln(x_i!)
$$

### 4.3 Solving by Defferentiation

Take derivative with respect to $\lambda$:
$$
\frac{d\ell}{d\lambda} = \frac{\sum_{i=1}^{n} x_i}{\lambda} - n
$$

Set it equal to 0:
$$
\frac{\sum x_i}{\lambda} - n = 0 \quad \Rightarrow \quad \hat{\lambda} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

### 4.4 Result Explanation

The MLE estimate turns out to be **sample mean**:
$$
\hat{\lambda} = \bar{x} = \frac{3+5+4+2+6}{5} = 4
$$

>For the Poisson distribution, the expectation is $E[X] = \lambda$, therefore using the sample mean to estimate the population expectation is entirely consistent with intuition.

### 4.5 Validate by Second Derivative (maximum)
$$
\frac{d^2\ell}{d\lambda^2} = -\frac{\sum x_i}{\lambda^2} \lt 0
$$

Since the second derivative is negative, this comfirms that we have found a **maximun point**.
