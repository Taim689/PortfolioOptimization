# ============================================================================
# Sensitivity test for exponential forgetting factor lambda.
# ============================================================================

# Load shared functions, constants, and plotting setup used by the sensitivity
# scripts.
include("sensitivity_common.jl")

# Load asset names, training returns, and test returns.
asset_names, R_train, R_test = load_return_data()

# Get the number of assets from the training return matrix.
n, _ = size(R_train)

# Start from an equally weighted portfolio.
x_start = fill(1.0 / n, n)

# Use the default proportional transaction cost rate for each asset.
c_vec = fill(DEFAULT_C_RATE, n)

# Forgetting factors to test.
# lambda = 1.00 gives no exponential forgetting.
# Lower values place more weight on recent observations.
LAMBDA_VALUES = [0.50, 0.70, 0.85, 0.90, 0.95, 0.97, 0.98, 0.99, 1.00]

# Store one performance row for each feasible test.
rows = NamedTuple[]

println("\nTesting exponential forgetting factor lambda")
println("="^90)

# Run the expanding window model once for each forgetting factor.
for lambda in LAMBDA_VALUES
    println("\nTesting lambda = $lambda")

    # Try to solve the expanding window backtest with the current lambda.
    #
    # feasible indicates whether the optimization succeeded.
    # ret contains realized portfolio returns.
    # W contains portfolio weights through time.
    # cost contains transaction costs through time.
    feasible, ret, W, cost =
        try_expanding(
            R_train,
            R_test,
            x_start,
            DEFAULT_MU_BAR,
            c_vec,
            DEFAULT_W_MAX,
            DEFAULT_USE_CAP,
            lambda
        )

    # Store performance metrics only if the test was feasible.
    if feasible
        push!(rows, performance_row("Expanding λ=$lambda", ret, W, cost))
    else
        println("lambda = $lambda failed.")
    end
end

# Convert collected results to a DataFrame for printing and plotting.
df = DataFrame(rows)

println("\nExpanding-window forgetting sensitivity")
show(df, allrows = true, allcols = true)
println()

# Apply the shared plotting style.
init_plot_style()

# Plot Sharpe ratio for each forgetting factor.
p1 = plot(
    df.Model,
    df.Sharpe;
    seriestype = :bar,
    title = "Forgetting-factor sensitivity: Sharpe ratio",
    xlabel = "Forgetting factor",
    ylabel = "Sharpe ratio",
    legend = false,
    xrotation = 35
)
display(p1)

# Plot total return for each forgetting factor.
p2 = plot(
    df.Model,
    df.TotalReturn_pct;
    seriestype = :bar,
    title = "Forgetting-factor sensitivity: total return",
    xlabel = "Forgetting factor",
    ylabel = "Total return (%)",
    legend = false,
    xrotation = 35
)
display(p2)

# Plot realized MAD for each forgetting factor.
p3 = plot(
    df.Model,
    df.RealizedMAD_pct;
    seriestype = :bar,
    title = "Forgetting-factor sensitivity: realized MAD",
    xlabel = "Forgetting factor",
    ylabel = "Realized MAD (%)",
    legend = false,
    xrotation = 35
)
display(p3)