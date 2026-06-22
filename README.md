# Portfolio Optimization with the Mean Absolute Deviation (MAD) Model

> Project work in Applied Mathematics (DTU, course 01666), Group 6.2 — Dániel László Seregi, Taim Kamal Brik and Victor Kragballe. Understanding how scenario-based risk measures perform under different portfolio constraints and rebalancing schemes is essential for evaluating the practical reliability of portfolio optimization models. The **Mean Absolute Deviation (MAD)** model serves as a linear alternative to the classical Markowitz variance model: it replaces quadratic risk with the average absolute deviation of portfolio returns, making it attractive from a computational point of view. This study uses monthly return data for twenty American ETFs from diverse sectors to construct a scenario matrix spanning various market regimes, and investigates whether adaptive rebalancing strategies — implemented through expanding and rolling estimation windows — improve out-of-sample performance relative to a static MAD portfolio. Allocation caps and proportional transaction costs are incorporated to reflect realistic investment constraints. Performance is evaluated primarily through the Sharpe ratio, supplemented by average monthly returns, realized MAD values and the HHI portfolio concentration metric. The analysis indicates that adaptive rebalancing strategies can enhance risk-adjusted performance relative to a static MAD portfolio: rolling and expanding window schemes show complementary improvements, with the rolling window yielding higher Sharpe ratios and the expanding window yielding higher total returns. Ultimately, **no single adaptive framework dominates unconditionally** — the preferred strategy depends purely on whether an investor prioritizes the Sharpe ratio, total returns or portfolio stability.

## 1. Introduction

Portfolio optimization addresses a fundamental problem faced by all exchange investors: financial markets are uncertain, asset returns fluctuate unpredictably, and choosing a portfolio by intuition alone often leads to unnecessarily risky outcomes. A systematic optimization framework provides a disciplined way to balance risk and return, quantify diversification, and construct portfolios that will hold against the winds of financial markets.

From a more mathematical standpoint, portfolio optimization is a quantitative investment strategy that aims to construct a portfolio by either maximizing expected return for a given level of risk or minimizing risk for a given level of expected return. Most classic models, such as the famous Markowitz mean-variance model, are formulated as quadratic programs (QP) because they measure portfolio risk using the variance of returns. A defining feature of this model is the covariance matrix, where each entry `σ_ij` measures how the returns of assets `i` and `j` move together, forming the mathematical basis for analyzing diversification. As Markowitz famously noted, diversification is the only free lunch in investing [^wiki]. Diversification reduces portfolio volatility because asset prices do not all move in the same direction or with the same magnitude. However, since asset returns are often inter-correlated, diversification can reduce but never fully eliminate variance.

Linear programming (LP) models such as the MAD model have become increasingly popular because LP solvers are generally more reliable, faster, and more scalable than QP solvers. LPs scale almost linearly with problem size, whereas QPs require costly matrix factorizations that grow cubically with the number of variables. Unlike the quadratic approach, which relies on estimating a covariance matrix that is sensitive to noise and estimation error, the MAD model uses a scenario-based strategy: historical returns are treated as discrete possible outcomes at the target time, which allows dependence between assets to be reflected directly through the empirical return distribution.

In this project we focus on the MAD model, which minimizes the average absolute deviation of portfolio returns from their expected value, incorporating both upward and downward fluctuations symmetrically. We begin by establishing a foundational MAD model with financial and heuristic constraints, after which we investigate whether different adaptive portfolio rebalancing methods can improve out-of-sample performance relative to a static portfolio strategy [^adaptive].

## 2. The MAD model

The MAD model computes the average absolute deviation of the portfolio's scenario returns `y_t`, weighted by the respective probability `p_t`, from the portfolio's expected return `μ`. Assuming all scenarios are equally likely, we set `p_t = 1/T` for each scenario `t`. The MAD measure accounts for all deviations of the rate of return of the portfolio from its expected value, both below and above the expected value [^book].

To transform the absolute deviation into a linear form, auxiliary variables `d_t` are introduced for each scenario, and the absolute value is represented using two linear inequality constraints. At the optimum the solver sets `d_t` as small as possible, so these constraints together make `d_t = |y_t − μ|`. The full linear program is:

```
minimize   Σ_t  p_t · d_t
subject to d_t ≥  (y_t − μ)          t = 1, …, T      (upper deviation)
           d_t ≥ −(y_t − μ)          t = 1, …, T      (lower deviation)
           y_t = Σ_j r_jt · x_j      t = 1, …, T      (scenario portfolio return)
           μ   = Σ_j μ_j · x_j                          (expected portfolio return)
           μ   ≥ μ₀                                     (minimum required return)
           Σ_j x_j = 1                                  (full investment)
           x_j ≥ 0                  j = 1, …, n         (no short-selling)
```

The primary advantage of the MAD model is that it can be formulated as a linear program. By using the auxiliary variables `d_t` as containers for the deviation in each scenario, the objective function and all constraints remain linear, allowing the optimization problem to be solved efficiently even with thousands of scenarios.

### Allocation constraints

To enhance the practical applicability of the MAD model, it is often necessary to incorporate "real features" that reflect specific investor preferences or market restrictions. If an investor wants to limit the weight of a specific asset to a maximum of e.g. 25%, the linear constraint `x_j ≤ 0.25` is added. The inclusion of allocation maximum constraints is a crucial tool for diversification enforcement. While the MAD model naturally seeks to reduce risk, the underlying historical data may sometimes suggest that a single asset has such a high expected return with low deviation that the model would otherwise allocate a disproportionately large share of the capital to it. By implementing this diversification constraint, the investor ensures that the portfolio remains diversified across at least several different ETFs — at a minimum 4 ETFs if the asset weight maximum is 25%. This prevents the model from putting all of the investor's eggs into one basket.

### Transaction costs

In practice, transaction costs are incurred when the composition of the portfolio is changed, so the cost should depend on the amount invested, not only on the final portfolio weights. We use a pure proportional cost structure (PPC), where the cost for asset `j` is proportional to the absolute change in its portfolio weight [^book]. To keep the model linear, the traded amount is split into two non-negative variables `b_j` and `s_j` (buying and selling), with `x_j − x_jᵒˡᵈ = b_j − s_j`. At the optimum `b_j + s_j` represents `|x_j − x_jᵒˡᵈ|`, so the total proportional transaction cost is `Σ_j c_j (b_j + s_j)`, which we account for by subtracting it from the expected portfolio return. We use a rolling implementation where `xᵒˡᵈ` is the portfolio obtained at the previous rebalancing date [^rebalancing].

We set the proportional cost rate to `c_j = 0.15%` for all assets. This is a simplified transaction cost assumption applied to the traded amount. ETF trading costs include bid-ask spreads and possible brokerage commissions, and the spread depends on factors such as liquidity, share price, volatility, and the cost of trading the underlying securities [^transcosts]. Since these costs differ across ETFs and over time, we use a fixed value of 0.15% in all experiments to keep the comparison between models consistent [^natixis][^schwab]. The transaction costs are only used in the adaptive models.

## 3. Data

To build our portfolio, we collect historical market data for 20 exchange traded funds (ETFs) from Yahoo Finance [^yahoo]. We choose ETFs because they can single-handedly provide broader market exposure across different sectors, regions and asset classes in comparison to traditional stocks, which makes them well-suited for studying diversification and portfolio optimization at the same time. Following the project scope, we use a historical period of around 11 years to capture different market conditions, including stable periods and times of increased market volatility. The data used in the project consists of historical price series for each ETF — the adjusted closing prices observed at the last day of each month, adjusted to account for stock splits, dividends, capital gains distributions and other corporate actions.

| ETF  | Sector | AR (%) | Age (yrs) | | ETF  | Sector | AR (%) | Age (yrs) |
|------|--------|-------:|----------:|-|------|--------|-------:|----------:|
| SPY  | Top 500 US companies         | 10.59 | 33 | | XLK  | S&P tech companies              | 9.51  | 27 |
| SCHG | US large growth stocks       | 15.65 | 16 | | AIRR | US industrial and regional banks| 16.57 | 12 |
| VUG  | Vanguard large growth        | 11.73 | 22 | | VGT  | Vanguard US tech stocks         | 13.93 | 22 |
| MGK  | Largest US growth stocks     | 12.96 | 18 | | SOXX | US semiconductor index          | 12.66 | 24 |
| XLY  | US consumer discretionary    | 9.56  | 27 | | GLD  | Gold                            | 11.35 | 21 |
| IYW  | US technology companies      | 8.49  | 25 | | XAR  | US aerospace defense            | 19.20 | 14 |
| PSI  | US semiconductor stocks      | 16.51 | 20 | | VPU  | US utility companies            | 10.12 | 22 |
| FDN  | US internet companies        | 13.39 | 19 | | XLV  | US healthcare companies         | 8.37  | 27 |
| IGM  | US tech and media stocks     | 11.73 | 25 | | PHO  | US water infrastructure         | 8.31  | 20 |
| FTEC | Fidelity US tech stocks      | 20.22 | 12 | | ARKW | Internet stocks                 | 19.42 | 11 |

*"AR" refers to the annualized return of each ETF.* We selected the ETFs based on two criteria: **maturity**, as all funds have been trading for more than 10 years, and **diversification** across economic sectors — by which we mean that the prices of the chosen ETFs are tied to distinct industries, thereby reducing overlap in their underlying market drivers. We have chosen only American ETFs as to isolate returns from foreign exchange noise.

**The scenario matrix.** The monthly return of asset `j` in month `t` is `r_jt = (q_jt − q_{j,t−1}) / q_{j,t−1}`, where `q_jt` is the adjusted closing price at the end of month `t`. The resulting returns are organized into a scenario matrix `R ∈ ℝ^{n×T}`, where each row corresponds to an ETF and each column corresponds to one monthly return scenario. Each scenario is treated as equally probable, `p_t = 1/T`. The earliest start date is determined by ARKW, which is the youngest and thereby most recently listed ETF (incepted 30 September 2014 [^arkw]), since all 20 ETFs must have valid adjusted closing prices in the same month before a common return matrix can be formed without missing values. Months with missing values for at least one ETF are removed. The final return matrix contains monthly returns from **October 2014 to December 2025**, giving **T = 135** monthly return observations.

**Train/test split.** The data is split into a training set and a test set using a 60/40 split. This leaves us with **`T_train` = 81** monthly observations to train on (October 2014 to June 2021) and **`T_test` = 54** monthly observations to test on (July 2021 to December 2025). This split is chosen to ensure that the training period is long enough to provide a reliable estimate of both the return and risk characteristics of the 20 ETFs, whilst still leaving the test period sufficiently long to evaluate performance across multiple market conditions.

## 4. Exploratory data analysis

Before applying the MAD model, we perform an exploratory analysis of the full dataset to understand the return and risk characteristics of the individual ETFs.

<p align="center">
  <img src="figures/scaled_prices.png" alt="Cumulative scaled prices for all 20 ETFs, log scale" width="90%">
</p>

*Figure 1. Cumulative scaled adjusted closing prices for all 20 ETFs, normalized to 1 at the start of October 2014.* The figure reveals significant dispersion in long-run performance: growth-oriented technology ETFs such as FTEC, SCHG and ARKW substantially outperform defensive sectors like VPU (utilities) and XLV (healthcare). The COVID-19 drawdown in early 2020 is visible as a sharp, synchronized drop across all ETFs, followed by a rapid recovery, followed by the broad market decline in 2022.

<p align="center">
  <img src="figures/monthly_returns.png" alt="Monthly returns for six selected ETFs" width="90%">
</p>

*Figure 2. Monthly returns for six selected ETFs over the full period (October 2014 – December 2025). Positive months are shown in blue and negative months in red. The red dashed line marks the sample mean return for each ETF.* The COVID-19 shock in March 2020 produced the largest single-month negative return across nearly all ETFs, while the recovery in April–May 2020 yielded the largest positive months. GLD exhibits noticeably lower variability than the equity ETFs, making it a natural risk-reducing component, while ARKW shows the widest swings in both directions.

<p align="center">
  <img src="figures/RiskVSsReturn.png" alt="Risk-return scatter of individual ETFs" width="60%">
</p>

*Figure 3. Risk-return scatter of individual ETFs over the full period (October 2014 – December 2025). The horizontal axis is the mean absolute deviation of monthly returns; the vertical axis is the mean monthly return.* Assets in the upper-left region are desirable, combining high return with low risk, while those in the lower right are unfavorable. SPY occupies the low-risk end of the spectrum with modest return, while SOXX, PSI and ARKW sit in the high-return and high-risk corner. No single ETF dominates in both dimensions simultaneously, which illustrates the motivation for portfolio optimization: by combining assets, the investor can achieve a risk-return profile that no individual ETF offers on its own.

<p align="center">
  <img src="figures/_efficient_frontier.png" alt="MAD efficient frontier" width="60%">
</p>

*Figure 4. MAD efficient frontier for the 20-ETF set computed over the training period with a 25% per-asset allocation cap. Each red dot is an optimal portfolio for a given required monthly return `μ₀`. Grey markers show the individual ETFs for reference.* The frontier lies to the left of all individual ETFs, which confirms that diversification does indeed reduce risk for all of the given levels of required return. As the portfolio is pushed toward the highest attainable return, the optimizer concentrates in the ETFs with the highest expected return subject to constraints, and the frontier flattens as no further increase in return is feasible.

## 5. MAD model extensions

The three MAD-based portfolio strategies compared in this project use out-of-sample performance on the test period. The models are estimated using only information available before each test month, and their realized performance is then evaluated on the subsequent test month. SPY is included as a passive benchmark over the same test period.

| Strategy | How it estimates | Rebalances? |
|---|---|:---:|
| **Non-adaptive** | The MAD model is solved once using the initial training period and held fixed throughout the test period | No |
| **Rolling window** | Re-optimized each test month using only the most recent `L` historical months (lookback window) | Yes |
| **Expanding window** | Re-optimized each test month using all available past data with exponentially declining weights | Yes |
| **SPY** | 100% SPY passive benchmark | No |

For the expanding model, since the estimation set expands over time, older observations can have less relevance for the current portfolio. To reduce this effect, the implementation uses exponential forgetting in the scenario probabilities: `p_t ∝ λ^(T−t)` with `0 < λ ≤ 1`. Recent observations receive higher weights, while older observations receive lower weights; when `λ = 1`, all scenarios have equal probability. For the rolling model, the lookback window is set to `L = 10` months for the reported results, but the parameter is adjustable in the code. Each model is tested one month at a time; the test month is used only afterward to measure realized performance.

## 6. Performance metrics

- **Sharpe ratio** [^sharpe] — the return of a portfolio relative to the amount of risk taken, `R̄_p / σ_p` (the risk-free rate is assumed to be zero in the numerical experiments). A higher Sharpe ratio indicates a more efficient portfolio with a better risk-return trade-off, and is the primary measure for comparing strategies.
- **Average monthly return** and **total return** — the realized profitability of the portfolio over the out-of-sample test period; a higher value indicates that the portfolio generated higher returns on average.
- **Realized MAD** — the average absolute deviation of the realized monthly portfolio returns from their sample mean in the test period. A lower MAD value indicates a more stable portfolio with less variation in returns.
- **MAD error** — the difference between the MAD value estimated by the model on the training data and the realized MAD on the test data.
- **HHI** (Herfindahl–Hirschman Index) [^hhi] — `Σ_j x_j²`, ranging in `[1/n, 1]`, where `1/n` corresponds to a perfectly diversified portfolio with equal weights and `1` corresponds to a fully concentrated portfolio. A lower HHI indicates a more diversified portfolio; an HHI close to 0.25 corresponds roughly to a portfolio concentrated in four assets at the cap.

## 7. Results

The parameter values used in the main comparison are `μ₀ = 1.9%` per month, `L = 10` for the rolling-window model, `λ = 0.97` for the expanding model, a transaction cost rate of 0.15%, and an allocation maximum of `w_max = 25%`. All schemes were solved using the **Gurobi** mathematical optimization solver [^gurobi].

| Model | Avg. return | Realized MAD | **Sharpe** | Model MAD | MAD error | HHI | Cost | **Total return** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Non-adaptive | 1.144% | 5.240% | 0.178 | 3.130% | −2.110% | 0.201 | — | 65.91% |
| **Rolling (L=10)** | 1.396% | 3.869% | **0.286** | 3.171% | −0.698% | 0.201 | 3.176% | 98.62% |
| **Expanding (λ=0.97)** | 1.500% | 4.933% | 0.239 | 4.314% | −0.619% | 0.207 | 1.064% | **101.68%** |
| SPY | 1.084% | 3.698% | 0.239 | — | — | 1.000 | — | 69.55% |

*Table 1. Out-of-sample performance over the 54-month test period.*

The best risk-adjusted performance is obtained by the rolling adaptive model, which has the highest Sharpe ratio of 0.286 — higher than both the non-adaptive model (0.178) and the expanding adaptive model (0.239). The rolling model also improves substantially on the non-adaptive strategy in terms of realized MAD, reducing it from 5.240% to 3.869% while increasing the average monthly return from 1.144% to 1.396%. This indicates that the rolling model is better able to adapt to changing market conditions during the test period, as expected, since the model only considers relatively recent data.

While all three models underestimate the realized risk during the test period, the adaptive models produce substantially smaller MAD errors than the non-adaptive MAD model. This indicates that incorporating new market information through rebalancing not only provides better returns, but also predicts risk more accurately.

The expanding adaptive model achieves the highest average monthly return (1.500%) and the highest total return (101.68%). However, this comes with a higher realized MAD of 4.933%, which reduces its Sharpe ratio to approximately the same level as SPY. The expanding model therefore performs well in terms of accumulated wealth, but it takes more realized risk than the rolling model. The passive SPY benchmark has lower realized MAD than the adaptive models, but also lower average return than both adaptive MAD strategies — the adaptive MAD models provide value relative to a passive market benchmark, especially when performance is evaluated using return per unit of realized volatility.

<p align="center">
  <img src="figures/cumulative_wealth.png" alt="Cumulative growth of $1 during the test period" width="80%">
</p>

*Figure 5. Cumulative growth of \$1 during the test period.* The adaptive models recover more strongly during the test period and finish above both the non-adaptive MAD model and SPY. The expanding model ends with the highest wealth index, while the rolling model follows closely behind. The target line based on `μ₀ = 1.9%` per month grows faster than all realized strategies, which shows that the required in-sample return target is ambitious relative to the realized out-of-sample performance.

<p align="center">
  <img src="figures/risk_return_tradeoff.png" alt="Risk-return trade-off during the test period" width="70%">
</p>

*Figure 6. Risk-return trade-off during the test period.* The rolling adaptive model gives the most attractive risk-return trade-off among the MAD strategies, because it increases average return while keeping realized MAD relatively low. The expanding model gives the highest return but also moves further to the right, meaning it has a higher realized MAD. The non-adaptive model performs worse than the rolling model, since it has both a lower average return and a higher realized MAD.

<p align="center">
  <img src="figures/headline_metrics.png" alt="Headline metrics bar charts" width="95%">
</p>

*Figure 7. Average monthly return, realized MAD, and Sharpe ratio for the MAD strategies and SPY.* The rolling model has the strongest Sharpe ratio, while the expanding model has the highest average monthly return. The non-adaptive model performs the weakest among the MAD strategies, mainly because it cannot adjust its portfolio after the initial training period.

<p align="center">
  <img src="figures/transaction_costs.png" alt="Cumulative transaction costs" width="48%">
  <img src="figures/hhi_over_time.png" alt="Portfolio concentration (HHI) over time" width="48%">
</p>

*Figure 8. Cumulative transaction costs (left) and portfolio concentration measured by HHI (right) during the test period.* The rolling model trades more aggressively than the expanding model, accumulating transaction costs of 3.176% over the test period compared with 1.064% for the expanding model — expected, since the rolling model only uses the most recent 10 months and reacts more strongly when recent return patterns change. Although the rolling model generates higher transaction costs, it still achieves the best risk-adjusted performance, suggesting that the benefits of faster adaptation outweigh the additional costs. The HHI values are mostly close to 0.20, which indicates that the portfolios are fairly concentrated and often close to the structure implied by the 25% allocation cap.

<p align="center">
  <img src="figures/final_weights.png" alt="Final portfolio weights at the end of the test period" width="90%">
</p>

*Figure 9. Final portfolio weights at the end of the test period.* The non-adaptive model remains fixed, while the rolling and expanding models end with different allocations due to their adaptive rebalancing rules. Several weights are close to the 25% cap, showing that the allocation maximum is active and affects the final portfolio composition. This confirms that the cap is important for preventing the optimizer from concentrating too heavily in a small number of ETFs.

## 8. Sensitivity analysis

A sensitivity analysis is conducted to examine how the adaptive MAD strategies respond to changes in the main model parameters. The purpose is to determine whether the main results are dependent on a single arbitrary parameter choice or whether the conclusions are stable across nearby specifications.

### Rolling-window length `L`

| L | Avg. return | MAD | Sharpe | HHI | Cost | Total return |
|---|---:|---:|---:|---:|---:|---:|
| 3  | 0.537% | 3.754% | 0.113 | 0.226 | 5.991% | 25.68% |
| 6  | 1.105% | 3.646% | 0.242 | 0.214 | 3.846% | 71.33% |
| **10** | 1.396% | 3.869% | **0.286** | 0.201 | 3.176% | 98.62% |
| 12 | 1.201% | 3.758% | 0.250 | 0.206 | 2.327% | 79.24% |
| 18 | 1.283% | 3.649% | 0.282 | 0.201 | 1.972% | 88.50% |
| **24** | 1.493% | 4.044% | **0.292** | 0.210 | 1.666% | 108.01% |
| 36 | 1.186% | 4.846% | 0.196 | 0.211 | 1.411% | 71.74% |
| 48 | 1.472% | 5.279% | 0.219 | 0.206 | 1.070% | 95.71% |
| 60 | 1.291% | 4.886% | 0.210 | 0.199 | 0.989% | 81.10% |

The rolling-window sensitivity shows that very short lookback windows perform poorly. With `L=3`, the model has a Sharpe ratio of only 0.113 and a total return of 25.68%, so a three-month window contains too little information and makes the optimizer overly sensitive to noise. The strongest Sharpe ratio is obtained at `L=24` (0.292, total return 108.01%), with the baseline `L=10` also strong (0.286, 98.62%). Longer windows such as `L=36`, `L=48` and `L=60` produce lower Sharpe ratios, mainly because realized MAD increases. This suggests that a moderate window length is preferable — short enough to adapt to changing market conditions, but long enough to avoid excessive noise.

<p align="center">
  <img src="figures/rolling_sharpe_sensitivity.png" alt="Rolling-window sensitivity: Sharpe ratio" width="48%">
  <img src="figures/rolling_mad_sensitivity.png" alt="Rolling-window sensitivity: realized MAD" width="48%">
</p>

*Figure 10. Rolling-window sensitivity measured by Sharpe ratio (left) and realized MAD (right).*

### Forgetting factor `λ`

| λ | Avg. return | MAD | Sharpe | HHI | Cost | Total return |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.909% | 3.650% | 0.202 | 0.220 | 4.364% | 54.54% |
| 0.70 | 0.811% | 3.710% | 0.172 | 0.209 | 3.889% | 45.82% |
| 0.85 | 1.193% | 3.992% | 0.235 | 0.207 | 2.971% | 77.17% |
| 0.90 | 1.305% | 3.992% | 0.256 | 0.206 | 2.428% | 88.07% |
| **0.95** | 1.635% | 4.764% | **0.267** | 0.210 | 1.466% | 117.60% |
| 0.97 | 1.500% | 4.933% | 0.239 | 0.207 | 1.064% | 101.68% |
| 0.98 | 1.488% | 5.402% | 0.219 | 0.209 | 0.903% | 96.92% |
| 0.99 | 1.212% | 5.523% | 0.175 | 0.203 | 0.602% | 69.08% |
| 1.00 | 1.025% | 5.954% | 0.137 | 0.210 | 0.539% | 49.89% |

The best expanding-window result is obtained around `λ=0.95`, where the Sharpe ratio is 0.267 and the total return is 117.60% — higher than the baseline `λ=0.97` (0.239, 101.68%). The results show that some forgetting is beneficial. When `λ` is too low (e.g. 0.50 or 0.70), the model becomes too reactive and transaction costs increase. When `λ` is too high, especially at 0.99 or 1.00, the model becomes too slow to adapt, realized MAD increases, and the Sharpe ratio falls. The expanding model therefore performs best when it gives substantial but not exclusive weight to recent observations.

<p align="center">
  <img src="figures/lambda_sharpe_sensitivity.png" alt="Forgetting-factor sensitivity: Sharpe ratio" width="48%">
  <img src="figures/lambda_mad_sensitivity.png" alt="Forgetting-factor sensitivity: realized MAD" width="48%">
</p>

*Figure 11. Forgetting-factor sensitivity measured by Sharpe ratio (left) and realized MAD (right).*

### Allocation maximum `w_max`

The allocation maximum was varied over `{5%, 7.5%, 10%, 15%, 20%, 25%, 35%, 50%, 100%}` for both adaptive models. Since there are 20 ETFs, the smallest feasible cap is 5%, which corresponds to an equal-weighted portfolio. At `w_max=5%`, both adaptive models are forced into the equal-weighted portfolio, so they have identical results and an HHI of 0.05. As the cap is relaxed, the models can take more concentrated positions and the HHI increases. For the rolling model, the best Sharpe ratios occur around `w_max=20%` and `w_max=25%` (0.288 and 0.286); relaxing the cap further does not improve performance, but instead makes the rolling model more concentrated and its total return declines. For the expanding model, relaxing the cap increases total return more strongly, reaching 126.17% at `w_max=100%` — but this also increases concentration (HHI rising to 0.363) and realized MAD (5.145%). The **25% cap is therefore a useful compromise**, as it allows the optimizer enough flexibility to improve performance while still limiting concentration to a few assets.

<p align="center">
  <img src="figures/wmax_sharpe_sensitivity.png" alt="Allocation-cap sensitivity: Sharpe ratio" width="55%">
</p>

*Figure 12. Allocation-cap sensitivity measured by Sharpe ratio.*

### Required monthly return `μ₀`

The required monthly return `μ₀` was varied over a grid from 0% to 5% in steps of 0.1 percentage points. For low values of `μ₀` the return constraint is weak or non-binding and the optimizer can focus primarily on minimizing MAD; as `μ₀` increases, the optimizer is forced toward portfolios with higher estimated return, which generally increases realized risk. For sufficiently high values of `μ₀`, some monthly optimization problems may become infeasible, in which case the adaptive implementation keeps the previous portfolio.

| Model | μ₀ | Avg. return | Realized MAD | Sharpe | Total return |
|---|---:|---:|---:|---:|---:|
| Non-adaptive | 0.0% | 1.145% | 3.038% | **0.317** | 78.72% |
| Non-adaptive | 1.9% | 1.144% | 5.240% | 0.178 | 65.91% |
| Non-adaptive | 2.4% | 1.344% | 6.665% | 0.161 | 71.57% |
| Rolling | 1.9% | 1.396% | 3.869% | 0.286 | 98.62% |
| **Rolling** | **2.0%** | 1.505% | 3.896% | **0.308** | 110.41% |
| Rolling | 3.1% | 1.547% | 4.141% | 0.297 | 113.46% |
| **Expanding** | **1.6%** | 1.511% | 4.069% | **0.296** | 110.06% |
| Expanding | 1.9% | 1.500% | 4.933% | 0.239 | 101.68% |

*Table 2. Selected results from the target-return sensitivity analysis.*

The non-adaptive model obtains its highest Sharpe ratio for low values of `μ₀`, where the return constraint is not restrictive. For `μ₀ = 0` the return constraint is not binding, so the models only minimize risk, and the non-adaptive model gives the highest out-of-sample Sharpe ratio (0.317), compared with 0.215 for the rolling model and 0.246 for the expanding model. This does not contradict the main results, since the MAD model is mainly used when the investor sets a required return. Once a positive target return is imposed — especially around `μ₀ = 1.6%` to `2.0%` — the adaptive models perform better because they can update their portfolios as new data become available. The rolling model performs best around `μ₀ = 2.0%` (Sharpe ≈ 0.308), while the expanding model performs best at an intermediate target, around `μ₀ = 1.6%`.

<p align="center">
  <img src="figures/mubar_sharpe.png" alt="Sharpe vs required return" width="32%">
  <img src="figures/mubar_totalreturn.png" alt="Total return vs required return" width="32%">
  <img src="figures/mubar_mad.png" alt="Realized MAD vs required return" width="32%">
</p>

*Figure 13. Sensitivity of the Sharpe ratio (left), total return (centre) and realized MAD (right) to the required monthly return `μ₀`.* For the non-adaptive model, realized MAD increases sharply as `μ₀` grows, indicating that the static model is forced into riskier portfolios when the required return is raised. The flat segments in the high-`μ₀` region of the expanding curve reflect that the implementation keeps the previous portfolio when the monthly optimization problem cannot be solved feasibly.

## 9. Discussion

The results show that adaptive rebalancing works best when the investor sets a positive return target. When `μ₀ = 0` the model mainly minimizes risk, and the static portfolio has the highest Sharpe ratio in the test period. For the main positive return targets tested, especially around `μ₀ = 1.6%` to `2.0%`, the adaptive models perform better because they can update the portfolio as new data becomes available.

The main motivation for introducing adaptive extensions was to address a limitation of the standard MAD model: in the non-adaptive formulation, portfolio weights are determined once using historical data and then kept fixed throughout the test period, which assumes that market conditions remain relatively stable — rarely the case in practice. The rolling and expanding approaches represent two different ways of using historical information. The rolling model focuses on the most recent observations and therefore adapts more quickly to changes in return patterns, which appears to improve the risk-adjusted performance. In contrast, the expanding model uses a larger information set and therefore produces more stable estimates, leading to higher total returns but also higher realized risk.

The allocation cap and transaction costs were introduced to make the model more realistic. Without an allocation cap, the optimizer tends to concentrate the portfolio in a small number of assets, so the cap serves as a diversification mechanism. Transaction costs are particularly important for adaptive strategies, since frequent rebalancing generates additional trading activity, and ignoring these costs would overestimate the practical value of adaptive portfolio management. The sensitivity analyses show that adaptive strategies do not automatically outperform the static model — performance depends strongly on the rolling window length `L`, the forgetting factor `λ` and the required return level, with moderate parameter values generally providing the best results.

## 10. Conclusion

This project investigated the practical performance of the MAD portfolio optimization model using monthly return data from twenty U.S. ETFs with an initial training period and an out-of-sample test period. The standard non-adaptive MAD model was extended with adaptive rebalancing schemes, allocation caps, and proportional transaction costs in order to better reflect realistic portfolio management principles.

The empirical results show that adaptive rebalancing can indeed improve out-of-sample performance relative to a static MAD portfolio when a positive target return is imposed. Both adaptive approaches achieved higher returns than the non-adaptive model, while the rolling adaptive model obtained the highest Sharpe ratio and the expanding adaptive model achieved the highest total return over the test period. Allocation caps reduced portfolio concentration and improved diversification, while transaction costs provided a more realistic assessment of the benefits of frequent rebalancing.

In conclusion, adaptive MAD models can provide meaningful improvements over a static MAD strategy, but **no single adaptive framework dominates under all evaluation criteria**. The preferred strategy depends solely on the investor's objective:

- **Strongest risk-adjusted performance** → the rolling model
- **Highest total return** → the expanding model
- **Lowest realized risk** → the passive SPY benchmark

Another key conclusion is that adaptive MAD portfolio optimization delivers strong performance only when supported by careful parameter selection and realistic modeling assumptions.

## 11. Repository layout and how to run

```
data/                          monthly_ROR_{train,test,full}.csv, mad_weights.csv
figures/                       result images referenced above
python/
  clean_returns_data_yf.ipynb  download + clean data, 60/40 split, export CSVs
  visualization.ipynb          scaled prices, return distributions, frontier, summaries
julia/
  MAD_solve.jl                 core MAD LP (simple + rebalancing with transaction costs)
  run_MAD_simple.jl            solve simple MAD, evaluate out-of-sample
  run_non_adaptive.jl          non-adaptive (static) strategy
  run_rolling_adaptive.jl      rolling-window strategy
  run_expanding_adaptive.jl    expanding-window strategy (exponential forgetting)
  main_comparison.jl           run all three + SPY, produce comparison figures
  sensitivity_common.jl        shared helpers for the sensitivity drivers
  driver_test_L.jl             sensitivity: rolling-window length
  driver_test_lambda.jl        sensitivity: forgetting factor
  driver_test_mubar.jl         sensitivity: required return
  driver_test_wmax.jl          sensitivity: allocation maximum
```

**1. Generate the data (Python)**

```bash
pip install yfinance pandas numpy matplotlib scipy
# run python/clean_returns_data_yf.ipynb -> writes data/monthly_ROR_{train,test,full}.csv
```

**2. Run the optimization (Julia)**

```bash
# requires a Gurobi license + Gurobi.jl, plus JuMP, CSV, DataFrames, Statistics, Plots
julia julia/main_comparison.jl      # main 3-way comparison + SPY benchmark
julia julia/driver_test_L.jl        # sensitivity sweeps
julia julia/driver_test_lambda.jl
julia julia/driver_test_mubar.jl
julia julia/driver_test_wmax.jl
```

> Gurobi can be swapped for the open-source **Cbc** solver in `MAD_solve.jl` (the import and `Model(Cbc.Optimizer)` lines are already present, commented out) if a Gurobi license isn't available.

The full code repository and the `G6.2_CodeAndData.zip` archive (the `.csv` files and selected Julia files) are available at: <https://github.com/Taim689/PortfolioOptimization>.

## References

[^book]: Mansini, R., Speranza, M.G., and Ogryczak, W. (2015). *Linear and Mixed Integer Programming for Portfolio Optimization*. Springer.

[^wiki]: Wikipedia. *Modern Portfolio Theory*. <https://en.wikipedia.org/wiki/Modern_portfolio_theory> (accessed Jun 16, 2026).

[^adaptive]: Rolfe, N. / Mcro Capital (Jan 2025). *Adaptive Investment Strategies: Navigating Dynamic Markets*. <https://www.mcroc.co.nz/blog/adaptive-investment-strategies-navigating-dynamic-markets> (accessed Jun 7, 2026).

[^sharpe]: Investopedia (Dec 2025). *Sharpe Ratio: Definition, Formula, and Examples*. <https://www.investopedia.com/terms/s/sharperatio.asp> (accessed Jun 9, 2026).

[^hhi]: Investopedia (Apr 2026). *Herfindahl–Hirschman Index (HHI): Definition, Formula, and Example*. <https://www.investopedia.com/terms/h/hhi.asp> (accessed Jun 16, 2026).

[^gurobi]: Gurobi Optimization, LLC. *Gurobi Optimizer Documentation and Resources*. <https://www.gurobi.com/> (accessed Jun 9, 2026).

[^yahoo]: Yahoo Finance. *Historical Market Data for ETFs*. <https://finance.yahoo.com/> (accessed Jun 9, 2026).

[^yfinance]: GeeksforGeeks (May 2025). *What is the yfinance library?* <https://www.geeksforgeeks.org/machine-learning/what-is-yfinance-library/> (accessed Jun 9, 2026).

[^arkw]: AAII. *ARK Next Generation Internet ETF (ARKW)*. <https://www.aaii.com/etf/ticker/ARKW> (accessed Jun 18, 2026).

[^natixis]: Natixis Investment Managers (Jan 2024). *Assessing ETF cost: Understanding the bid/ask spread*. <https://www.im.natixis.com/en-us/insights/portfolio-construction/2024/etf-cost-bid-ask-spread> (accessed Jun 7, 2026).

[^schwab]: Charles Schwab (Feb 2025). *ETFs: Expense Ratios and Other Costs*. <https://www.schwab.com/learn/story/etfs-how-much-do-they-really-cost> (accessed Jun 7, 2026).

[^rebalancing]: Investopedia (Mar 2025). *How to Rebalance Your Portfolio*. <https://www.investopedia.com/how-to-rebalance-your-portfolio-7973806> (accessed Jun 16, 2026).

[^transcosts]: Investopedia (Oct 2025). *Understanding Transaction Costs: Definition, Examples, and Impact*. <https://www.investopedia.com/terms/t/transactioncosts.asp> (accessed Jun 16, 2026).
