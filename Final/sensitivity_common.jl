# ============================================================================
# Shared helpers for sensitivity tests.
# ============================================================================

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

const TRAIN_FILE = "data/monthly_ROR_train.csv"
const TEST_FILE  = "data/monthly_ROR_test.csv"

const DEFAULT_MU_BAR = 0.019
const DEFAULT_L = 10
const DEFAULT_LAMBDA = 0.97
const DEFAULT_C_RATE = 0.0015
const DEFAULT_W_MAX = 0.25
const DEFAULT_USE_CAP = true

function load_return_data()
    df_train = CSV.read(TRAIN_FILE, DataFrame)
    df_test  = CSV.read(TEST_FILE, DataFrame)

    asset_names_train = String.(df_train[:, 1])
    asset_names_test  = String.(df_test[:, 1])

    if asset_names_train != asset_names_test
        error("Train and test files have different asset order.")
    end

    asset_names = asset_names_train

    R_train = Matrix{Float64}(df_train[:, 2:end])
    R_test  = Matrix{Float64}(df_test[:, 2:end])

    n, _ = size(R_train)
    n_test, _ = size(R_test)

    if n_test != n
        error("Train and test files have different number of assets.")
    end

    return asset_names, R_train, R_test
end

realized_mad(r) = mean(abs.(r .- mean(r)))

function sharpe(r)
    s = std(r)
    return s <= 1e-12 ? NaN : mean(r) / s
end

hhi_series(W) = [sum(@view(W[:, k]).^2) for k in 1:size(W, 2)]

function wealth_path(r)
    w = ones(length(r) + 1)

    @inbounds for k in eachindex(r)
        w[k + 1] = w[k] * (1.0 + r[k])
    end

    return w
end

function performance_row(model_name, r, W, costs)
    return (
        Model = model_name,
        AvgReturn_pct = 100 * mean(r),
        RealizedMAD_pct = 100 * realized_mad(r),
        Sharpe = sharpe(r),
        AvgHHI = mean(hhi_series(W)),
        TotalCost_pct = 100 * sum(costs),
        TotalReturn_pct = 100 * (wealth_path(r)[end] - 1.0)
    )
end

function try_non_adaptive(R_train, R_test, mu_bar, asset_names)
    try
        ret, W, cost, x_static =
            RunNonAdaptive.run_non_adaptive(R_train, R_test, mu_bar, asset_names)

        return true, ret, W, cost
    catch err
        @warn "Non-adaptive failed" mu_bar error = err
        return false, nothing, nothing, nothing
    end
end

function try_rolling(R_train, R_test, x_start, mu_bar, c_vec, w_max, use_cap, L)
    try
        ret, W, cost =
            RunRollingAdaptive.run_rolling_adaptive(
                R_train,
                R_test,
                x_start,
                mu_bar,
                c_vec,
                w_max,
                use_cap,
                L
            )

        return true, ret, W, cost
    catch err
        @warn "Rolling adaptive failed" mu_bar L w_max error = err
        return false, nothing, nothing, nothing
    end
end

function try_expanding(R_train, R_test, x_start, mu_bar, c_vec, w_max, use_cap, lambda)
    try
        ret, W, cost =
            RunExpandingAdaptive.run_expanding_adaptive(
                R_train,
                R_test,
                x_start,
                mu_bar,
                c_vec,
                w_max,
                use_cap,
                lambda
            )

        return true, ret, W, cost
    catch err
        @warn "Expanding adaptive failed" mu_bar lambda w_max error = err
        return false, nothing, nothing, nothing
    end
end

function init_plot_style()
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
end