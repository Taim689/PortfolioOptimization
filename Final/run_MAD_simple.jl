using CSV
using DataFrames
using Statistics
using JuMP
using Gurobi
using MathOptInterface
include("MAD_solve.jl")
using .MAD_solve

const MOI = MathOptInterface

  
# Parameters
train_file = "data/monthly_ROR_train.csv"
test_file  = "data/monthly_ROR_test.csv"

mu_bar = 0.012  # minimum required monthly return (in-sample)
  
# Load data
df_train = CSV.read(train_file, DataFrame)
df_test  = CSV.read(test_file,  DataFrame)

asset_names = String.(df_train[:, 1])

R_train = Matrix{Float64}(df_train[:, 2:end])
R_test  = Matrix{Float64}(df_test[:,  2:end])

n,      T_train = size(R_train)
n_test, K       = size(R_test)

println("Assets:          ", n)
println("Training months: ", T_train)
println("Test months:     ", K)

  
# Solve MAD model on training data
  

# Equal probability for each training scenario
p = fill(1.0 / T_train, T_train)

# Expected return of each asset over the training period
mu_assets = [sum(p[t] * R_train[j, t] for t in 1:T_train) for j in 1:n]

model, x_var, y_var, d_var, mu_var =
    MAD_solve.solve_MAD_simple(mu_bar, R_train, p, mu_assets, asset_names, false)

println("\nTermination status: ", termination_status(model))

  
# Results
  

if termination_status(model) == MOI.OPTIMAL

    weights = value.(x_var)

    println("In-sample MAD:         ", round(100 * objective_value(model), digits = 4), " %")
    println("In-sample mean return: ", round(100 * value(mu_var),          digits = 4), " %")

    println("\nSelected assets:")
    for j in 1:n
        if weights[j] > 1e-8
            println("  ", rpad(asset_names[j], 10), round(weights[j], digits = 4))
        end
    end

    # Out-of-sample evaluation
    returns_test    = [sum(R_test[j, k] * weights[j] for j in 1:n) for k in 1:K]
    avg_return_test = mean(returns_test)
    mad_test        = mean(abs.(returns_test .- avg_return_test))
    sharpe_test     = std(returns_test) > 1e-12 ? avg_return_test / std(returns_test) : NaN

    println("\nOut-of-sample performance:")
    println("  Average monthly return: ", round(100 * avg_return_test, digits = 4), " %")
    println("  Realized MAD:           ", round(100 * mad_test,        digits = 4), " %")
    println("  Sharpe ratio:           ", round(sharpe_test,            digits = 4))

    weights_df = DataFrame(ETF = asset_names, Weight = weights)
    CSV.write("data/mad_weights.csv", weights_df) # save for visualization in python
    println("\nWeights saved to data/mad_weights.csv")

else
    println("No optimal solution found.")
end