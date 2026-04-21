using CSV
using DataFrames
using Statistics
using JuMP
using Cbc
using MathOptInterface
const MOI = MathOptInterface

# Read CSV file
csv_file = "monthly_ROR_2010_2020.csv"
df = CSV.read(csv_file, DataFrame)

# First column contains dates
dates = df[:, 1]

# Asset names are all columns except the first
asset_names = String.(names(df)[2:end])

# Return matrix:
# rows = scenarios / time periods
# columns = assets
R = Matrix{Float64}(df[:, 2:end])
# T = number of scenarios
# n = number of assets
T, n = size(R)
# Mean return of each asset across all scenarios
mu_assets = vec(mean(R, dims = 1))
# Equal scenario probabilities
p = fill(1.0 / T, T)
# Example old portfolio:
# Here we assume the current portfolio is equally weighted.
# In a real rebalancing setting, this should be replaced by
# the actual portfolio weights from the previous period.
x_old = fill(1.0 / n, n)

# Proportional transaction costs for each asset
# 0.0015 = 0.15%
c_vec = fill(0.0015, n)

function solve_MAD_rebalancing(mu_bar, R, p, mu_assets, x_old, c_vec, asset_names)
    T, n = size(R)

    println("Number of scenarios: ", T)
    println("Number of assets: ", n)

    model = Model(Cbc.Optimizer)

    # New portfolio weights after rebalancing
    @variable(model, x[1:n] >= 0)

    # Portfolio return in each scenario
    @variable(model, y[1:T])

    # Absolute deviations from mean return
    @variable(model, d[1:T] >= 0)

    # Mean portfolio return
    @variable(model, mu)

    # u[j] = increase in weight of asset j
    # v[j] = decrease in weight of asset j
    #
    # Together they represent turnover:
    # x[j] - x_old[j] = u[j] - v[j]
    # and at optimum:
    # |x[j] - x_old[j]| = u[j] + v[j]
    @variable(model, u[1:n] >= 0)
    @variable(model, v[1:n] >= 0)

    # Objective: minimize mean absolute deviation
    @objective(model, Min, sum(p[t] * d[t] for t in 1:T))

    # d[t] >= y[t] - mu
    @constraint(model, [t in 1:T], d[t] >= y[t] - mu)

    # d[t] >= -(y[t] - mu)
    @constraint(model, [t in 1:T], d[t] >= -(y[t] - mu))

    # Scenario portfolio return:
    # y[t] = sum_j R[t,j] * x[j]
    @constraint(model, [t in 1:T], y[t] == sum(R[t, j] * x[j] for j in 1:n))

    # Mean portfolio return:
    # mu = sum_j mu_assets[j] * x[j]
    @constraint(model, mu == sum(mu_assets[j] * x[j] for j in 1:n))

    # Rebalancing relation:
    # new weight - old weight = buy part - sell part
    @constraint(model, [j in 1:n], x[j] - x_old[j] == u[j] - v[j])

    # Minimum required return after proportional transaction costs
    #
    # Transaction cost for asset j:
    # c_vec[j] * |x[j] - x_old[j]|
    #
    # Since |x[j] - x_old[j]| = u[j] + v[j],
    # total proportional transaction cost is:
    # sum_j c_vec[j] * (u[j] + v[j])
    @constraint(model,
        mu - sum(c_vec[j] * (u[j] + v[j]) for j in 1:n) >= mu_bar
    )

    # Full investment
    @constraint(model, sum(x[j] for j in 1:n) == 1)

    # Limit max weight in any single asset to 25%
    @constraint(model, [j in 1:n], x[j] <= 0.25)

    optimize!(model)

    return model, x, mu, u, v
end

# Target return per period
mu_bar = 0.019

model, x, mu, u, v = solve_MAD_rebalancing(mu_bar, R, p, mu_assets, x_old, c_vec, asset_names)

println("Termination status: ", termination_status(model))

if termination_status(model) == MOI.OPTIMAL
    println("Objective value (MAD): ", objective_value(model))
    println("Mean portfolio return: ", value(mu))

    weights = value.(x)
    buys = value.(u)
    sells = value.(v)

    # Total turnover = sum of absolute changes in weights
    turnover = sum(buys[j] + sells[j] for j in 1:n)

    # Total proportional transaction cost
    tc_total = sum(c_vec[j] * (buys[j] + sells[j]) for j in 1:n)

    println("Total turnover: ", turnover)
    println("Total transaction cost: ", tc_total)
    println("Net mean return after transaction costs: ", value(mu) - tc_total)

    println("\nOptimal portfolio weights:")
    for j in 1:n
        println(rpad(asset_names[j], 10), "  ", round(weights[j], digits = 6))
    end

    println("\nRebalancing details:")
    for j in 1:n
        if buys[j] > 1e-8 || sells[j] > 1e-8
            println(
                rpad(asset_names[j], 10),
                " old = ", round(x_old[j], digits = 6),
                "  new = ", round(weights[j], digits = 6),
                "  buy = ", round(buys[j], digits = 6),
                "  sell = ", round(sells[j], digits = 6)
            )
        end
    end

    println("\nPortfolio mean recomputed from weights: ",
        sum(mu_assets[j] * value(x[j]) for j in 1:n)
    )
else
    println("No optimal solution found.")
end