# Down-and-Out Barrier Option Pricing via Monte Carlo

> **Personal project**, done independently outside coursework.

Pricing of a **Down-and-Out Put** (barrier option) in the Black-Scholes model, with a focus on the discretization bias of the barrier crossing and its correction via the **Brownian Bridge** technique.

---

## Products

### Vanilla European Put
$$P^{\text{euro}} = e^{-rT}\mathbb{E}\left[(K - S(T))^+\right]$$

### Down-and-Out Put (discrete monitoring)
$$P^{DO,\Delta} = e^{-rT}\mathbb{E}\left[(K - S(T))^+\,\mathbf{1}_{\min_{T_i} S(T_i)\,\geq\, B}\right]$$

### Down-and-In Put
$$P^{DI} = e^{-rT}\mathbb{E}\left[(K - S(T))^+\mathbf{1}_{\min_{u\leq T} S(u)\,\leq\, B}\right]$$

**Put-Call-Barrier parity:**
$$P^{DO} + P^{DI} = P^{\text{euro}}$$

---

## Key idea: Brownian Bridge correction

Discrete monitoring misses barrier crossings between observation dates → **upward bias** on survival probability. The correction uses the conditional probability of crossing $B$ between two consecutive observations:

$$\pi_i = \exp\left(\frac{-2\ln(S_{t_i}/B)\ln(S_{t_{i+1}}/B)}{\sigma^2\*\Delta t}\right)$$

The continuous survival probability is then approximated as:

$$\hat{\zeta}^{DO} = \prod_{i}\(1 - \pi_i)$$

This **eliminates the discretization bias** even with a coarse time grid.

---

## Model

Black-Scholes in dimension 1:

$$dS(t) = S(t)\left(r\*dt + \sigma\*dW(t)\right), \quad S(0) = S_0 > 0$$

$$S(t) = S_0\exp\left(\left(r - \tfrac{1}{2}\sigma^2\right)t + \sigma W(t)\right)$$

---

## Parameters

| Parameter | Value |
|-----------|-------|
| `S₀` | 1 |
| `K` | 1 |
| `B` | 0.7 |
| `σ` | 0.15 |
| `r` | 0.015 |
| `T` | 2 years |
| `Δt` | 1/52 (weekly) |

> All simulations use **Uniform random variables only** via Box-Muller transform.

---

## Topics covered

| Q | Content |
|---|---------|
| Q1 | Closed-form solution for S(t) |
| Q2 | Analytical Black-Scholes put formula |
| Q3–4 | MC estimator for P_euro + 90% CI |
| Q5 | Analytical P_DO when B ≥ K |
| Q6 | Discrete MC estimator for P_DO,Δ |
| Q7 | Antithetic variables + convergence comparison |
| Q8 | P_DO,Δ vs barrier level B ∈ [0.5, 1] |
| Q9 | P_DO,Δ vs volatility σ for S₀ = 1 and S₀ = 0.8 |
| Q10 | Brownian Bridge correction for continuous barrier |
| Q11 | P_DO,Δ vs monitoring frequency Δ |
| Q12 | Survival probability ζ: discrete vs BB-corrected vs theoretical |
| Q13 | Put-Call-Barrier parity: P_DO + P_DI = P_euro |
| Q14 | Control variate via P_DI, variance reduction vs B |

---

## Key results

**Effect of B on variance reduction (Q14):**

- B small: almost all paths survive → Var(P_DI) ≈ 0 → control variate very effective
- B close to K: paths split evenly between knock-out and survival → no variance reduction gain

**Discretization bias (Q12):**

- Discrete monitoring **overestimates** survival probability (misses intra-period crossings)
- Brownian Bridge correction converges to the theoretical value for all values of Δ

---

## Run

```bash
pip install numpy matplotlib
python barrier_option_commented.py
```

## Dependencies

`numpy` · `matplotlib`
