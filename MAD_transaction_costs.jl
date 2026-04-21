module MAD_transaction_costs

using CSV
using DataFrames
using Statistics
using JuMP
using Cbc

export solve_MAD_transaction

function solve_MAD_transaction(mu_bar, R, p, mu_assets, asset_names, c_vec, f_vec, l_vec, u_vec)
    T, n = size(R)
    println( "Number of scenarios: ", T)
    println( "Number of assets: ", n)
    model = Model(Cbc.Optimizer)

    #Portfolio weights
    @variable(model, x[1:n] >= 0)

    #Portfolio return in each scenario
    @variable(model, y[1:T])

    #Absolute deviations from mean return
    @variable(model, d[1:T] >= 0)

    #Mean portfolio return
    @variable(model, mu)

    #Binary variable for whether asset j is included
    @variable(model, z[1:n], Bin)

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
    @constraint(model, mu - sum(c_vec[j] * x[j] + f_vec[j] * z[j] for j in 1:n) >= mu_bar)

    # Full investment
    @constraint(model, sum(x[j] for j in 1:n) == 1)

    # Minimum and maximum weight if asset j is selected
    @constraint(model, [j in 1:n], l_vec[j] * z[j] <= x[j])
    @constraint(model, [j in 1:n], x[j] <= u_vec[j] * z[j])

    optimize!(model)

    return model, x, mu, z
    end
end

#= 

println("new run -------------------------------------------------------------------------------------")
csv_file = "monthly_ROR_2010_2020.csv"
df = CSV.read(csv_file, DataFrame)

dates = df[:, 1]
asset_names = String.(names(df)[2:end])
println("Asset names: ", asset_names)
R = Matrix{Float64}(df[:, 2:end])

T, n = size(R)
mu_assets = vec(mean(R, dims=1))
p = fill(1.0 / T, T)

mu_bar = 0.019
c_vec = fill(0.00015, n)
f_vec = fill(0.0000001, n)
l_vec = fill(0.0, n)
u_vec = fill(0.25, n)

model, x, mu, z = solve_MAD(mu_bar, R, p, mu_assets, asset_names, c_vec, f_vec, l_vec, u_vec)

println("Termination status: ", termination_status(model))

if termination_status(model) == MOI.OPTIMAL
    println("Objective value (MAD): ", objective_value(model))
    println("Mean portfolio return: ", value(mu))

    weights = value.(x)
    selected = value.(z)

    weights_df = DataFrame(
    Asset = asset_names,
    Weight = weights)

    #CSV.write("trained_weights.csv", weights_df)
    #println("Weights saved to trained_weights.csv")

    println("Net mean return after transaction costs: ",
        value(mu) - sum(c_vec[j] * weights[j] + f_vec[j] * selected[j] for j in 1:n)
    )

    println("\nOptimal portfolio weights:")
    for j in 1:n
        println(rpad(asset_names[j], 10), "  ", round(weights[j], digits=6))
    end

    println("Portfolio mean recomputed from weights: ",
        sum(mu_assets[j] * value(x[j]) for j in 1:n))

    println("\nSelected assets:")
    for j in 1:n
        if value(x[j]) > 1e-8
            println(rpad(asset_names[j], 10),
                    "  x = ", round(value(x[j]), digits=5))
        end
    end
else
    println("No optimal solution found.")
end

  =#