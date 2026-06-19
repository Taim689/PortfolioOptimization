module RunRollingAdaptive

using JuMP
using MathOptInterface
using Statistics
include("MAD_solve.jl")
using .MAD_solve

const MOI = MathOptInterface

function run_rolling_adaptive(
    R_train::Matrix{Float64},
    R_test::Matrix{Float64},
    x_old_start::Vector{Float64},
    mu_bar::Float64,
    c_vec::Vector{Float64},
    w_max::Float64,
    use_allocation_max::Bool,
    rolling_lookback::Int
)
    n = size(R_train, 1)
    K = size(R_test, 2)

    R_hist = copy(R_train)
    x_old  = copy(x_old_start)

    returns = zeros(K)
    costs   = zeros(K)
    weights = zeros(n, K)

    for k in 1:K
        # Use the most recent rolling lookback months from the history.
        T_hist    = size(R_hist, 2)
        first_col = max(1, T_hist - rolling_lookback + 1)
        R_window  = R_hist[:, first_col:T_hist]

        T_window   = size(R_window, 2)
        p          = fill(1.0 / T_window, T_window)
        mu_assets  = [sum(p[t] * R_window[j, t] for t in 1:T_window) for j in 1:n]

        model, x_var, _, _, _, _, _ =
            MAD_solve.solve_MAD_rebalancing(
                R_window,
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
        # Append the realized test month to the history.
        R_hist = hcat(R_hist, R_test[:, k])
    end

    return returns, weights, costs
end

end 
