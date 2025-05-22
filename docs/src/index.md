# TrustRegionRadius.jl Documentation

## Algorithms 
```@docs
trust_region_with_cg
gradient_descent
LS_steepest_backtrack
```

## Parameters structures

### Conn Gould and Toint parameters

The trust-region radius is updated according to the following rule:

```math
\begin{equation}
    \Delta_{k+1} \in 
        \begin{cases}
        [\gamma_1 \Delta_k, \gamma_2 \Delta_k], & \text{if } \rho_k < \eta_1, \\
        [\gamma_2 \Delta_k, \Delta_k], & \text{if } \eta_1 \leq \rho_k < \eta_2, \\
        [\Delta_k, +\infty), & \text{if } \rho_k \geq \eta_2.
        \end{cases}
\end{equation}
```
where $0 < \gamma_1 \leq \gamma_2 < 1 < \gamma_3$ and $0 < \eta_1 \leq \eta_2 < 1$ are constants.

```@docs
TointGouldTointParameters
```
The trust-region radius can also be updated as follows:

```math
\begin{equation}
    \Delta_{k+1} =
    \begin{cases}
    \gamma^{-1} \, \Delta_k, & \text{if } \rho_k < \eta_1, \\
    \gamma \, \Delta_k, & \text{if } \rho_k \geq \eta_1,
    \end{cases}
\end{equation}
```
for $\gamma > 1$ and $0 < \eta_1 < 1$.

```@docs
SimpleTointGouldTointParameters
```
### Scheinberg parameters
If $\rho_k \geq \eta_1$, then set $x_{k+1} = x_k + s_k$ and

```math
\begin{equation}
    \Delta_{k+1} =
        \begin{cases}
        \gamma^{-1} \Delta_k, & \text{if } \|g_k\| < \eta_2 \Delta_k, \\
        \min\{\gamma \Delta_k, \Delta_{\max}\}, & \text{if } \|g_k\| \geq \eta_2 \Delta_k.
        \end{cases}
\end{equation}
```
where $\Delta_{\max} > 0$ is a constant.

Otherwise, set $x_{k+1} = x_k$ and $\Delta_{k+1} = \gamma^{-1} \Delta_k$.

```@docs
SimpleScheinbergParameters
ScheinbergParameters
```
### Yuan and Fan parameters
Another update strategy is based on the gradient norm:

```math
\begin{equation}
    \Delta_k = \mu_k \| \nabla f(x_k) \|^\alpha,
\end{equation}
```
with $\alpha \in (0,1]$, and $\mu_k > 0$ possibly adapted per iteration:

```math
\begin{equation}
    \mu_{k+1} =
        \begin{cases}
        c_1 \mu_k, & \text{if } \rho_k < \eta_2, \\
        c_2 \mu_k, & \text{if } \rho_k \geq \eta_2 \text{ and } \|s_k\| > \frac{1}{2} \Delta_k, \\
        \mu_k, & \text{otherwise},
        \end{cases}
\end{equation}
```
where $0 < c_1 < 1 < c_2$, $\eta_2 \in (0, 1)$.

```@docs
YuanFanParameters
```
### Hei parameters
The trust-region radius can also be updated using a non-decreasing function $R_\eta$ as follows:

```math
\begin{equation}
    \Delta_{k+1} = R_\eta(\rho_k) \|s_k\|,
\end{equation}
```
where $R_\eta : \mathbb{R} \to \mathbb{R}_+$ is a **non-decreasing** function satisfying:

1.  
    ``\lim_{t \to -\infty} R_\eta(t) = \beta, \quad \text{where } \beta \in [0,1) \text{ is a small constant};``
2.  For all ``t < \eta``, it holds that
    ``R_\eta(t) \leq 1 - \gamma_1, \quad \text{where } \gamma_1 \in (0, 1 - \beta) \text{ is a constant};``
3.  At the parameter value ``t = \eta``, we have
    ``
    R_\eta(\eta) = 1 + \gamma_2, \quad \text{where } \gamma_2 \in (0, +\infty) \text{ is a constant};
    ``
4.  
    ``
    \lim_{t \to +\infty} R_\eta(t) = M, \quad \text{where } M \in (1 + \gamma_2, +\infty) \text{ is a constant}.
    ``

A possible exponential form for $R_\eta$ is:
```math
\begin{equation}
R_{\eta}(t, \beta, \gamma_1, \gamma_2, M, \lambda_1, \lambda_2) =
\begin{cases}
    \beta + (1 - \gamma_1 - \beta) e^{\lambda_1 (t - \eta)}, & t < \eta, \\
    1 + \gamma_2 + (M - (1 + \gamma_2)) \big(1 - e^{-\lambda_2 (t - \eta)}\big), & t \geq \eta.
\end{cases}
\end{equation}
```
where $\lambda_1, \lambda_2 > 0$ and the parameters $\beta, \gamma_1, \gamma_2, M$ are constants that can be set by the user.

```@docs
HeiParameters
```

```math
\begin{equation}
    \Delta_{k+1} = R_\eta(\rho_k) \|g_k\|,
\end{equation}
```

```@docs
HeiGradParameters
```
### Hei and Fan parameters
The trust-region radius can also be updated using a non-decreasing function $R_\eta$ as follows:
```math
\begin{equation}
    \Delta_k = \mu_k \| \nabla f(x_k) \|^\alpha,
\end{equation}
```
with $\alpha \in (0,1]$, and $\mu_k > 0$ possibly adapted per iteration:
```math
\begin{equation}
    \mu_{k+1} = R_\eta(\rho_k) \mu_k.
\end{equation}
```

```@docs
HeiFanYuanParameters
```

## Stopping criteria
```@docs
StoppingCriteriaGradient
```

## Algorithm information
```@docs   
AlgorithmInfoTR
```


