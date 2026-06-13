module RunExpandingAdaptive

using JuMP
using MathOptInterface
using Statistics
include("MAD_solve.jl")
using .MAD_solve
const MOI = MathOptInterface

"""
    run_expanding_adaptive(R_train, R_test, x_old_start, mu_bar,
                           c_vec, w_max, use_allocation_max, lambda)

Re-solve the MAD rebalancing model before every test month using an ever-growing
history of observations.  Older observations are down-weighted by exponential
forgetting with decay factor `lambda`.

Arguments
---------
- `R_train`            : (n × T_train) training return matrix.
- `R_test`             : (n × K) test return matrix.
- `x_old_start`        : Length-n vector of initial portfolio weights before month 1.
- `mu_bar`             : Minimum required monthly return (scalar).
- `c_vec`              : Length-n vector of proportional transaction cost rates.
- `w_max`              : Maximum allocation per asset (scalar, e.g. 0.25).
- `use_allocation_max` : Bool — whether to enforce the allocation cap.
- `lambda`             : Exponential forgetting factor in (0, 1].  lambda = 1 gives
                         equal weights; smaller values discount older observations more.

Returns
-------
- `returns` : Length-K vector of realized net-of-cost portfolio returns.
- `weights` : (n × K) matrix of portfolio weights chosen before each test month.
- `costs`   : Length-K vector of transaction costs paid each month.
"""
function run_expanding_adaptive(
    R_train::Matrix{Float64},
    R_test::Matrix{Float64},
    x_old_start::Vector{Float64},
    mu_bar::Float64,
    c_vec::Vector{Float64},
    w_max::Float64,
    use_allocation_max::Bool,
    lambda::Float64
)
    n = size(R_train, 1)
    K = size(R_test, 2)

    R_hist = copy(R_train)
    x_old  = copy(x_old_start)

    returns = zeros(K)
    costs   = zeros(K)
    weights = zeros(n, K)

    for k in 1:K
        T_hist = size(R_hist, 2)

        # Exponential forgetting: recent scenarios receive higher probability.
        raw_weights = [lambda^(T_hist - t) for t in 1:T_hist]
        p           = raw_weights ./ sum(raw_weights)
        mu_assets   = [sum(p[t] * R_hist[j, t] for t in 1:T_hist) for j in 1:n]

        model, x_var, _, _, _, _, _ =
            MAD_solve.solve_MAD_rebalancing(
                R_hist,
                p,
                mu_assets,
                mu_bar,
                x_old,
                c_vec,
                w_max,
                use_allocation_max
            )

        if termination_status(model) == MOI.OPTIMAL
            x_new    = value.(x_var)
            costs[k] = sum(c_vec[j] * abs(x_new[j] - x_old[j]) for j in 1:n)
        else
            x_new    = copy(x_old)
            costs[k] = 0.0
        end

        # Realized return minus transaction cost paid this month.
        returns[k]    = sum(R_test[j, k] * x_new[j] for j in 1:n) - costs[k]
        weights[:, k] = x_new

        x_old  = copy(x_new)
        # Expand the history with the now-observed test month.
        R_hist = hcat(R_hist, R_test[:, k])
    end

    return returns, weights, costs
end

end # module
