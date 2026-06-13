using CSV
using DataFrames
using Statistics
using Printf
using JuMP
using MathOptInterface
using Plots
using LinearAlgebra: dot

include("MAD_solve.jl");              using .MAD_solve
include("run_non_adaptive.jl");       using .RunNonAdaptive
include("run_rolling_adaptive.jl");   using .RunRollingAdaptive
include("run_expanding_adaptive.jl"); using .RunExpandingAdaptive

const MOI = MathOptInterface

  
# Parameters
  

const TRAIN_FILE = "data/monthly_ROR_train.csv"
const TEST_FILE  = "data/monthly_ROR_test.csv"

const MU_BAR   = 0.019    # minimum required in-sample monthly return, 1.9% per month
const L        = 10       # rolling lookback window length in months
const LAMBDA   = 0.97     # exponential forgetting factor
const C_RATE   = 0.0015   # proportional transaction cost rate, 0.15%
const W_MAX    = 0.25     # maximum allocation per asset
const USE_CAP  = true     # allocation cap is used in all MAD models

mkpath("data")

  
# Load data
# CSV layout: column 1 = ticker, remaining columns = monthly returns
  

df_train = CSV.read(TRAIN_FILE, DataFrame)
df_test  = CSV.read(TEST_FILE, DataFrame)

asset_names_train = String.(df_train[:, 1])
asset_names_test  = String.(df_test[:, 1])

asset_names = asset_names_train

R_train = Matrix{Float64}(df_train[:, 2:end])   # n × T_train
R_test  = Matrix{Float64}(df_test[:, 2:end])    # n × K

n, T_train = size(R_train)
n_test, K  = size(R_test)

x_start = fill(1.0 / n, n)
c_vec = fill(C_RATE, n)

@printf("Assets: %d | Train months: %d | Test months: %d\n", n, T_train, K)
@printf(
    "mu_bar = %.2f%%/month | L = %d | lambda = %.2f | cost = %.3f%% | cap = %.0f%%\n",
    100 * MU_BAR,
    L,
    LAMBDA,
    100 * C_RATE,
    100 * W_MAX
)

  
# Run the three MAD Models

println("\nSolving non-adaptive MAD ...")
ret_na, W_na, cost_na, x_static =
    RunNonAdaptive.run_non_adaptive(R_train, R_test, MU_BAR, asset_names)

println("Solving rolling-window adaptive MAD ...")
ret_rl, W_rl, cost_rl =
    RunRollingAdaptive.run_rolling_adaptive(
        R_train,
        R_test,
        x_start,
        MU_BAR,
        c_vec,
        W_MAX,
        USE_CAP,
        L
    )

println("Solving expanding adaptive MAD with exponential forgetting ...")
ret_ex, W_ex, cost_ex =
    RunExpandingAdaptive.run_expanding_adaptive(
        R_train,
        R_test,
        x_start,
        MU_BAR,
        c_vec,
        W_MAX,
        USE_CAP,
        LAMBDA
    )

  
# SPY reference (100% SPY)
idx_spy = findfirst(==("SPY"), asset_names)
ret_spy = vec(R_test[idx_spy, :])
W_spy = zeros(n, K)
W_spy[idx_spy, :] .= 1.0
cost_spy = zeros(K)

names_mad = [
    "Non-adaptive",
    "Rolling (L=$L)",
    "Expanding (lambda=$LAMBDA)"
]

names_mad_tex = [
    "Non-adaptive",
    "Rolling (\$L=$L\$)",
    "Expanding (\$\\lambda=$LAMBDA\$)"
]

rets_mad  = [ret_na, ret_rl, ret_ex]
Ws_mad    = [W_na, W_rl, W_ex]
costs_mad = [cost_na, cost_rl, cost_ex]

names_all = [
    "Non-adaptive",
    "Rolling (L=$L)",
    "Expanding (lambda=$LAMBDA)",
    "SPY"
]

rets_all  = [ret_na, ret_rl, ret_ex, ret_spy]
Ws_all    = [W_na, W_rl, W_ex, W_spy]
costs_all = [cost_na, cost_rl, cost_ex, cost_spy]

colors_mad = [:steelblue, :darkorange, :seagreen]
colors_all = [:steelblue, :darkorange, :seagreen, :grey40]

M_mad = length(rets_mad)
M_all = length(rets_all)

  
# Metric helpers
  

realized_mad(r) = mean(abs.(r .- mean(r)))

sharpe(r) = begin
    s = std(r)
    s <= 1e-12 ? NaN : mean(r) / s
end

hhi_series(W) = [sum(@view(W[:, k]).^2) for k in 1:size(W, 2)]

function wealth_path(r)
    w = ones(length(r) + 1)

    @inbounds for k in eachindex(r)
        w[k + 1] = w[k] * (1.0 + r[k])
    end

    return w
end

function insample_mad(x, Rwin, p)
    Tt = size(Rwin, 2)
    y = [dot(@view(Rwin[:, t]), x) for t in 1:Tt]
    mu = sum(p[t] * y[t] for t in 1:Tt)

    return sum(p[t] * abs(y[t] - mu) for t in 1:Tt)
end

function model_mad(idx)
    if idx == 1
        p = fill(1.0 / T_train, T_train)
        return insample_mad(x_static, R_train, p)
    end

    vals = zeros(K)

    for k in 1:K
        hist = k == 1 ? R_train : hcat(R_train, R_test[:, 1:k-1])
        Th = size(hist, 2)

        if idx == 2
            first_col = max(1, Th - L + 1)
            win = hist[:, first_col:Th]
            p = fill(1.0 / size(win, 2), size(win, 2))
            vals[k] = insample_mad(@view(Ws_mad[2][:, k]), win, p)
        elseif idx == 3
            weights = [LAMBDA^(Th - t) for t in 1:Th]
            p = weights ./ sum(weights)
            vals[k] = insample_mad(@view(Ws_mad[3][:, k]), hist, p)
        else
            error("model_mad is only defined for the three MAD strategies.")
        end
    end

    return mean(vals)
end

  
# Build performance metrics
  

avg_ret_all  = mean.(rets_all)
rel_mad_all  = realized_mad.(rets_all)
sharpe_all   = sharpe.(rets_all)
avg_hhi_all  = [mean(hhi_series(W)) for W in Ws_all]
tot_cost_all = sum.(costs_all)
tot_ret_all  = [wealth_path(r)[end] - 1.0 for r in rets_all]

mad_model_mad = [model_mad(i) for i in 1:M_mad]
mad_error_mad = mad_model_mad .- rel_mad_all[1:M_mad]

model_mad_all = vcat(mad_model_mad, NaN)
mad_error_all = vcat(mad_error_mad, NaN)

summary_df = DataFrame(
    Model = names_all,
    AvgReturn_pct = round.(100 .* avg_ret_all, digits = 3),
    RealizedMAD_pct = round.(100 .* rel_mad_all, digits = 3),
    Sharpe = round.(sharpe_all, digits = 3),
    ModelMAD_pct = round.(100 .* model_mad_all, digits = 3),
    MADerror_pct = round.(100 .* mad_error_all, digits = 3),
    AvgHHI = round.(avg_hhi_all, digits = 3),
    TotalCost_pct = round.(100 .* tot_cost_all, digits = 3),
    TotalReturn_pct = round.(100 .* tot_ret_all, digits = 2)
)

  
# Console output
println("\n", "="^110)
println("OUT-OF-SAMPLE PERFORMANCE")
println("="^110)

@printf(
    "%-24s %9s %9s %8s %9s %9s %8s %9s %9s\n",
    "Model",
    "AvgRet%",
    "MAD%",
    "Sharpe",
    "mMAD%",
    "MADerr%",
    "HHI",
    "Cost%",
    "TotRet%"
)

println("-"^110)

for i in 1:M_all
    if i <= M_mad
        @printf(
            "%-24s %9.3f %9.3f %8.3f %9.3f %9.3f %8.3f %9.3f %9.2f\n",
            names_all[i],
            100 * avg_ret_all[i],
            100 * rel_mad_all[i],
            sharpe_all[i],
            100 * model_mad_all[i],
            100 * mad_error_all[i],
            avg_hhi_all[i],
            100 * tot_cost_all[i],
            100 * tot_ret_all[i]
        )
    else
        @printf(
            "%-24s %9.3f %9.3f %8.3f %9s %9s %8.3f %9.3f %9.2f\n",
            names_all[i],
            100 * avg_ret_all[i],
            100 * rel_mad_all[i],
            sharpe_all[i],
            "--",
            "--",
            avg_hhi_all[i],
            100 * tot_cost_all[i],
            100 * tot_ret_all[i]
        )
    end
end

println("="^110)

best = argmax(sharpe_all)
@printf("\nHighest Sharpe ratio: %s (%.3f).\n", names_all[best], sharpe_all[best])


# Global plot style
  

gr()

default(
    titlefontsize = 17,
    guidefontsize = 15,
    tickfontsize = 12,
    legendfontsize = 12,
    framestyle = :box,
    grid = true,
    gridalpha = 0.25,
    size = (1000, 640),
    dpi = 300,
    left_margin = 9Plots.mm,
    right_margin = 6Plots.mm,
    top_margin = 5Plots.mm,
    bottom_margin = 9Plots.mm
)

months = 1:K

  
# Figure 1: cumulative wealth
  

p1 = plot(
    title = "Cumulative growth of \$1 during the test period",
    xlabel = "Test month",
    ylabel = "Wealth index",
    legend = :topleft
)

for i in 1:M_all
    w = wealth_path(rets_all[i])
    plot!(
        p1,
        0:K,
        w;
        lw = 3,
        color = colors_all[i],
        label = @sprintf("%s  (×%.2f)", names_all[i], w[end])
    )
end

plot!(
    p1,
    0:K,
    [(1.0 + MU_BAR)^k for k in 0:K];
    lw = 2,
    ls = :dot,
    color = :black,
    label = @sprintf("Target (%.1f%%/month)", 100 * MU_BAR)
)

display(p1)

  
# Figure 2: risk return scatter
  

xvals = 100 .* rel_mad_all
yvals = 100 .* avg_ret_all

xpad = 0.18 * (maximum(xvals) - minimum(xvals)) + 0.05
ypad = 0.22 * (maximum(yvals) - minimum(yvals)) + 0.05

p2 = scatter(
    title = "Risk-return trade-off during the test period",
    xlabel = "Realized MAD (%)",
    ylabel = "Average monthly return (%)",
    legend = :outerright,
    size = (1180, 600),
    xlims = (minimum(xvals) - xpad, maximum(xvals) + xpad),
    ylims = (minimum(yvals) - ypad, maximum(yvals) + ypad)
)

for i in 1:M_all
    scatter!(
        p2,
        [xvals[i]],
        [yvals[i]];
        markersize = 14,
        color = colors_all[i],
        markerstrokewidth = 1.5,
        markerstrokecolor = :white,
        label = @sprintf("%s  (Sharpe %.2f)", names_all[i], sharpe_all[i])
    )
end

display(p2)

  
# Figure 3: headline metrics
  

function bar_with_labels(labels, values, ttl, ylab, bar_colors)
    b = bar(
        labels,
        values;
        title = ttl,
        color = bar_colors,
        legend = false,
        xrotation = 18,
        ylabel = ylab
    )

    vmax = maximum(values)
    vmin = min(0.0, minimum(values))
    ylims!(b, vmin, vmax * 1.20)

    for i in 1:length(values)
        annotate!(b, i, values[i] + 0.04 * vmax, text(@sprintf("%.2f", values[i]), 11))
    end

    return b
end

b1 = bar_with_labels(names_all, 100 .* avg_ret_all, "Avg monthly return (%)", "%", colors_all)
b2 = bar_with_labels(names_all, 100 .* rel_mad_all, "Realized MAD (%)", "%", colors_all)
b3 = bar_with_labels(names_all, sharpe_all, "Sharpe ratio", "", colors_all)

p3 = plot(
    b1,
    b2,
    b3;
    layout = (1, 3),
    size = (1780, 580),
    top_margin = 7Plots.mm,
    bottom_margin = 16Plots.mm,
    left_margin = 6Plots.mm
)

display(p3)

  
# Figure 4: portfolio concentration over time
  
p4 = plot(
    title = "Portfolio concentration over time (HHI)",
    xlabel = "Test month",
    ylabel = "HHI (sum of squared weights)",
    legend = :bottomright
)

for i in 1:M_mad
    plot!(
        p4,
        months,
        hhi_series(Ws_mad[i]);
        lw = 3,
        color = colors_mad[i],
        label = names_mad[i]
    )
end

hline!(
    p4,
    [1.0 / n];
    lw = 2,
    ls = :dot,
    color = :black,
    label = @sprintf("Equal weight (1/%d)", n)
)

hline!(
    p4,
    [1.0 / round(Int, 1.0 / W_MAX)];
    lw = 2,
    ls = :dash,
    color = :grey50,
    label = @sprintf("Cap reference (1/%d)", round(Int, 1.0 / W_MAX))
)

display(p4)

  
# Figure 5: cumulative transaction costs
  
# Only adaptive MAD strategies have monthly rebalancing costs.

p5 = plot(
    title = "Cumulative transaction cost during the test period",
    xlabel = "Test month",
    ylabel = "Cumulative cost (%)",
    legend = :topleft
)

for i in 2:M_mad
    plot!(
        p5,
        months,
        100 .* cumsum(costs_mad[i]);
        lw = 3,
        color = colors_mad[i],
        label = names_mad[i]
    )
end

display(p5)
  
# Figure 6: final portfolio weights
  
# SPY is excluded here because its portfolio is mechanically 100% SPY.
# The purpose of this figure is to compare the final allocations of the MAD models.

w_final = [100 .* Ws_mad[i][:, end] for i in 1:M_mad]

active = [
    j for j in 1:n
    if maximum(w_final[i][j] for i in 1:M_mad) > 0.5
]

xa = collect(1:length(active))
bw = 0.26

p6 = plot(
    title = "Final portfolio weights at the end of the test period",
    xlabel = "ETF",
    ylabel = "Weight (%)",
    legend = :topright,
    size = (1250, 640),
    bottom_margin = 14Plots.mm
)

for i in 1:M_mad
    offset = (i - (M_mad + 1) / 2) * bw

    bar!(
        p6,
        xa .+ offset,
        [w_final[i][j] for j in active];
        bar_width = bw,
        color = colors_mad[i],
        linecolor = :white,
        label = names_mad[i]
    )
end

hline!(
    p6,
    [100 * W_MAX];
    lw = 2,
    ls = :dash,
    color = :black,
    label = @sprintf("Cap %.0f%%", 100 * W_MAX)
)

plot!(
    p6;
    xticks = (xa, asset_names[active]),
    xrotation = 45
)

display(p6)