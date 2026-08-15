
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

> Clearly $\hat\theta$ remains changed, since $ln(x)$ is a monotonically increasing function.


## 3. Example: Parameter Estimation of the Normal Distribution

Let $X \sim \mathcal{N}(\mu, \sigma^2)$

