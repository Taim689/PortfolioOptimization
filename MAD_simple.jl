using CSV
using DataFrames
using Statistics
using JuMP
using Cbc

csv_file = "weekly_returns.csv"
df = CSV.read(csv_file, DataFrame)

dates = df[:, 1]
asset_names = String.(names(df)[2:end])
R = Matrix{Float64}(df[:, 2:end])


# T = number of scenarios
# n = number of assets
T, n = size(R)
println( "Number of scenarios: ", T)
println( "Number of assets: ", n)
mu_assets = vec(mean(R, dims=1))
p = fill(1.0 / T, T)
println(p)
mu_bar = 0.003

println("Asset mean returns:")
for j in 1:n
    println(asset_names[j], "  ", round(mu_assets[j], digits=6))
end
println("\nMaximum asset mean return: ", round(maximum(mu_assets), digits=6))
println("Chosen mu_bar: ", mu_bar)

#Mad model
model = Model(Cbc.Optimizer)

#Portfolio weights
@variable(model, x[1:n] >= 0)

#Portfolio return in each scenario
@variable(model, y[1:T])

#Absolute deviations from mean return
@variable(model, d[1:T] >= 0)

#Mean portfolio return
@variable(model, mu)

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

# Minimum required return
@constraint(model, mu >= mu_bar)

# Full investment
@constraint(model, sum(x[j] for j in 1:n) == 1)

#limit max weight in any single asset to 60%
@constraint(model, [j in 1:n], x[j] <= 0.6)

optimize!(model)
println("Termination status: ", termination_status(model))
println("Objective value (MAD): ", objective_value(model))
println("Mean portfolio return: ", value(mu))

weights = value.(x)

println("\nOptimal portfolio weights:")
for j in 1:n
    println(rpad(asset_names[j], 10), "  ", round(weights[j], digits=6))
end

scenario_returns = value.(y)
deviations = value.(d)

