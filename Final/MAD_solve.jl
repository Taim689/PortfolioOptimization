module MAD_solve

using JuMP
using Cbc
using Gurobi

export solve_MAD_simple, solve_MAD_rebalancing

function solve_MAD_simple(mu_bar, R, p, mu_assets, asset_names, use_allocation_max= true, w_max=0.25)
    n, T = size(R)
    
    println("Number of scenarios: ", T)
    println("Number of assets: ", n)
    
    #model = Model(Cbc.Optimizer)
    model = Model(Gurobi.Optimizer)
    
    # Portfolio weights
    @variable(model, x[1:n] >= 0)
    
    # Portfolio return in each scenario
    @variable(model, y[1:T])
    
    # Absolute deviations from mean return
    @variable(model, d[1:T])
    
    # Mean portfolio return
    @variable(model, mu)
    
    # Objective: minimize mean absolute deviation
    @objective(model, Min, sum(p[t] * d[t] for t in 1:T))
    
    # d[t] >= y[t] - mu
    @constraint(model, [t in 1:T], d[t] >= y[t] - mu)
    
    # d[t] >= -(y[t] - mu)
    @constraint(model, [t in 1:T], d[t] >= -(y[t] - mu))
    
    # Scenario portfolio return:
    # y[t] = sum_j R[j,t] * x[j]
    @constraint(model, [t in 1:T], y[t] == sum(R[j, t] * x[j] for j in 1:n))
    
    # Mean portfolio return:
    # mu = sum_j mu_assets[j] * x[j]
    @constraint(model, mu == sum(mu_assets[j] * x[j] for j in 1:n))
    
    # Minimum required return
    @constraint(model, mu >= mu_bar)
    
    # Full investment
    @constraint(model, sum(x[j] for j in 1:n) == 1)
    
    # Limit max weight in any single asset to w_max 
    if use_allocation_max
        @constraint(model, [j in 1:n], x[j] <= w_max)
    end
    optimize!(model)

    return model, x, y, d, mu
end

end

function solve_MAD_rebalancing(R, p, mu_assets, mu_bar, x_old, c_vec, w_max, use_allocation_max)
    n, T = size(R)

    # another model can be used here
    #model = Model(Cbc.Optimizer)

    model = Model(Gurobi.Optimizer)
    
    # Portfolio weights
    @variable(model, x[1:n] >= 0)
    
    # Portfolio return in each scenario
    @variable(model, y[1:T])
    
    # Absolute deviations from mean return
    @variable(model, d[1:T])
    
    # Mean portfolio return
    @variable(model, mu)
    
    # Buy variables
    @variable(model, b[1:n] >= 0)
    
    # Sell variables
    @variable(model, s[1:n] >= 0)
    
    # Objective: minimize mean absolute deviation
    @objective(model, Min, sum(p[t] * d[t] for t in 1:T))
    
    # d[t] >= y[t] - mu
    @constraint(model, [t in 1:T], d[t] >= y[t] - mu)
    
    # d[t] >= -(y[t] - mu)
    @constraint(model, [t in 1:T], d[t] >= -(y[t] - mu))
    
    # Scenario portfolio return:
    # y[t] = sum_j R[j,t] * x[j]
    @constraint(model, [t in 1:T], y[t] == sum(R[j, t] * x[j] for j in 1:n))
    
    # Mean portfolio return:
    # mu = sum_j mu_assets[j] * x[j]
    @constraint(model, mu == sum(mu_assets[j] * x[j] for j in 1:n))
    
    # Rebalancing constraint:
    # x[j] - x_old[j] = b[j] - s[j]
    @constraint(model, [j in 1:n], x[j] - x_old[j] == b[j] - s[j])
    
    # Minimum required return after transaction costs
    @constraint(model,
        mu - sum(c_vec[j] * (b[j] + s[j]) for j in 1:n) >= mu_bar)
    
    # Full investment
    @constraint(model, sum(x[j] for j in 1:n) == 1)
    
    # Limit max weight in any single asset to w_max
    if use_allocation_max
        @constraint(model, [j in 1:n], x[j] <= w_max)
    end
    optimize!(model)
    return model, x, y, d, mu, b, s
end
