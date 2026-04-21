using CSV
using DataFrames
using Statistics
using JuMP
using Cbc
using MathOptInterface
const MOI = MathOptInterface

csv_file = "monthly_ROR_2010_2020.csv"
df = CSV.read(csv_file, DataFrame)

dates = df[:, 1]
asset_names = String.(names(df)[2:end])

# If returns in the CSV are percentages like 2.5, divide by 100.
# If they are already decimals like 0.025, remove the ./ 100 line.
R = Matrix{Float64}(df[:, 2:end])

T, n = size(R)
mu_assets = vec(mean(R, dims = 1))
p = fill(1.0 / T, T)

function solve_MAD_with_prop_costs(mu_bar, R, p, mu_assets, c_vec)
    T, n = size(R)

    @assert length(p) == T "Length of p must equal number of scenarios T"
    @assert length(mu_assets) == n "Length of mu_assets must equal number of assets n"
    @assert length(c_vec) == n "Length of c_vec must equal number of assets n"

    model = Model(Cbc.Optimizer)

    @variable(model, x[1:n] >= 0)
    @variable(model, y[1:T])
    @variable(model, d[1:T] >= 0)
    @variable(model, mu)

    @objective(model, Min, sum(p[t] * d[t] for t in 1:T))

    @constraint(model, [t in 1:T], d[t] >= y[t] - mu)
    @constraint(model, [t in 1:T], d[t] >= -(y[t] - mu))

    @constraint(model, [t in 1:T], y[t] == sum(R[t, j] * x[j] for j in 1:n))
    @constraint(model, mu == sum(mu_assets[j] * x[j] for j in 1:n))

    @constraint(model, mu - sum(c_vec[j] * x[j] for j in 1:n) >= mu_bar)
    
    @constraint(model, sum(x[j] for j in 1:n) == 1)

    #limit max weight in any single asset to 25%
    @constraint(model, [j in 1:n], x[j] <= 0.25)

    optimize!(model)

    return model, x, mu, y, d
end


mu_bar = 0.019
# all assets have the same proportional cost of 0.015% (0.00015 in decimal)
c_vec = fill(0.00015, n)

model, x, mu, y, d = solve_MAD_with_prop_costs(mu_bar, R, p, mu_assets, c_vec)

println("Termination status: ", termination_status(model))

if termination_status(model) == MOI.OPTIMAL
    weights = value.(x)

    println("Objective value (MAD): ", objective_value(model))
    println("Mean portfolio return: ", value(mu))
    println("Net mean return after transaction costs: ",
        value(mu) - sum(c_vec[j] * weights[j] for j in 1:n)
    )

    println("\nOptimal portfolio weights:")
    for j in 1:n
        println(rpad(asset_names[j], 10), "  ", round(weights[j], digits = 6))
    end
else
    println("No optimal solution found.")
end