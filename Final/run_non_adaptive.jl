module RunNonAdaptive

using JuMP
using MathOptInterface
using Statistics
include("MAD_solve.jl")
using .MAD_solve

const MOI = MathOptInterface

"""
    run_non_adaptive(R_train, R_test, mu_bar, asset_names)

Solve the simple MAD model once on the training data and evaluate the fixed portfolio on the test data.

Arguments
---------
- `R_train`      : (n × T_train) matrix of monthly returns for the training period.
- `R_test`       : (n × K) matrix of monthly returns for the test period.
- `mu_bar`       : Minimum required monthly return (scalar).
- `asset_names`  : Vector of n asset ticker strings.

Returns
-------
- `returns`  : Vector of length K with realized portfolio returns each test month.
- `weights`  : (n × K) matrix where every column equals the static weight vector.
- `costs`    : Vector of length K of zeros (no rebalancing costs).
- `x_static` : Vector of length n with the optimal portfolio weights.
"""
function run_non_adaptive(
    R_train::Matrix{Float64},
    R_test::Matrix{Float64},
    mu_bar::Float64,
    asset_names::Vector{String}
)
    n, T_train = size(R_train)
    K          = size(R_test, 2)

    # Equal scenario probabilities over the training period.
    p_train        = fill(1.0 / T_train, T_train)
    mu_assets_train = [sum(p_train[t] * R_train[j, t] for t in 1:T_train) for j in 1:n]

    model_static, x_static_var, _, _, _ =
        MAD_solve.solve_MAD_simple(
            mu_bar,
            R_train,
            p_train,
            mu_assets_train,
            asset_names
        )

    if termination_status(model_static) != MOI.OPTIMAL
        error("Non-adaptive model did not solve to optimality.")
    end

    x_static = value.(x_static_var)

    # Evaluate the fixed portfolio on every test month.
    returns = [sum(R_test[j, k] * x_static[j] for j in 1:n) for k in 1:K]
    weights = repeat(x_static, 1, K)
    costs   = zeros(K)

    return returns, weights, costs, x_static
end

end # module
