# ============================================================================
# Sensitivity test for rolling window length L.
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

# Rolling window lengths to test.
# Each value defines how many past observations are used when estimating
# expected returns and MAD during the rolling optimization.
L_VALUES = [3, 6, 10, 12, 18, 24, 36, 48, 60]

# Store one performance row for each feasible rolling window test.
rows = NamedTuple[]

println("\nTesting rolling-window length L")
println("="^90)

# Run the rolling model once for each selected window length.
for L in L_VALUES
    println("\nTesting L = $L")

    # Try to solve the rolling backtest with the current window length.
    #
    # feasible indicates whether the rolling optimization succeeded.
    # ret contains realized portfolio returns.
    # W contains portfolio weights through time.
    # cost contains transaction costs through time.
    feasible, ret, W, cost =
        try_rolling(
            R_train,
            R_test,
            x_start,
            DEFAULT_MU_BAR,
            c_vec,
            DEFAULT_W_MAX,
            DEFAULT_USE_CAP,
            L
        )

    # Store performance metrics only if the test was feasible.
    if feasible
        push!(rows, performance_row("Rolling L=$L", ret, W, cost))
    else
        println("L = $L failed.")
    end
end

# Convert collected results to a DataFrame for printing and plotting.
df = DataFrame(rows)

println("\nRolling-window sensitivity")
show(df, allrows = true, allcols = true)
println()

# Apply the shared plotting style.
init_plot_style()

# Plot Sharpe ratio for each rolling window length.
p1 = plot(
    df.Model,
    df.Sharpe;
    seriestype = :bar,
    title = "Rolling-window sensitivity: Sharpe ratio",
    xlabel = "Rolling-window length",
    ylabel = "Sharpe ratio",
    legend = false,
    xrotation = 35
)
display(p1)

# Plot total return for each rolling window length.
p2 = plot(
    df.Model,
    df.TotalReturn_pct;
    seriestype = :bar,
    title = "Rolling-window sensitivity: total return",
    xlabel = "Rolling-window length",
    ylabel = "Total return (%)",
    legend = false,
    xrotation = 35
)
display(p2)

# Plot realized MAD for each rolling window length.
p3 = plot(
    df.Model,
    df.RealizedMAD_pct;
    seriestype = :bar,
    title = "Rolling-window sensitivity: realized MAD",
    xlabel = "Rolling-window length",
    ylabel = "Realized MAD (%)",
    legend = false,
    xrotation = 35
)
display(p3)