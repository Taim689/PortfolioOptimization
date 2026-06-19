# ============================================================================
# Sensitivity test for allocation maximum W_MAX.
# ============================================================================

# Load shared functions, constants, data loading, and plotting setup.
include("sensitivity_common.jl")

# Load asset names, training returns, and test returns.
asset_names, R_train, R_test = load_return_data()

# Get the number of assets from the training return matrix.
n, _ = size(R_train)

# Start the adaptive models from an equally weighted portfolio.
x_start = fill(1.0 / n, n)

# Use the default proportional transaction cost rate for each asset.
c_vec = fill(DEFAULT_C_RATE, n)

# Allocation caps to test.
# A value of 0.05 means that each asset can receive at most 5% of the portfolio.
# A value of 1.00 means that there is effectively no per asset cap.
W_MAX_VALUES = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 1.00]

# Store one performance row for each feasible model and allocation cap.
rows = NamedTuple[]

println("\nTesting allocation maximum W_MAX")
println("="^90)

# Test each allocation cap.
for w_max in W_MAX_VALUES

    # If the cap is smaller than 1/n, the budget cannot be fully allocated
    # across n assets while respecting the cap.
    if w_max < 1.0 / n
        println("Skipping W_MAX = $w_max because it is below 1/n and cannot be feasible.")
        continue
    end

    println("\nTesting W_MAX = $w_max")

 
    # Test the rolling model with the current allocation cap.
 
    feasible_rl, ret_rl, W_rl, cost_rl =
        try_rolling(
            R_train,
            R_test,
            x_start,
            DEFAULT_MU_BAR,
            c_vec,
            w_max,
            DEFAULT_USE_CAP,
            DEFAULT_L
        )

    # If feasible, store the resulting performance metrics.
    if feasible_rl
        push!(
            rows,
            performance_row(
                "Rolling W=$(round(100 * w_max, digits = 1))%",
                ret_rl,
                W_rl,
                cost_rl
            )
        )
    end

 
    # Test the expanding model with the current allocation cap.
 
    feasible_ex, ret_ex, W_ex, cost_ex =
        try_expanding(
            R_train,
            R_test,
            x_start,
            DEFAULT_MU_BAR,
            c_vec,
            w_max,
            DEFAULT_USE_CAP,
            DEFAULT_LAMBDA
        )

    # If feasible, store the resulting performance metrics.
    if feasible_ex
        push!(
            rows,
            performance_row(
                "Expanding W=$(round(100 * w_max, digits = 1))%",
                ret_ex,
                W_ex,
                cost_ex
            )
        )
    end
end

# Convert collected results to a DataFrame for printing and plotting.
df = DataFrame(rows)

println("\nAllocation-cap sensitivity")
show(df, allrows = true, allcols = true)
println()

# Apply the shared plotting style.
init_plot_style()

# Extract the allocation cap labels from the model names.
# Example: "Rolling W=25.0%" becomes "25.0%".
caps = unique(last.(split.(df.Model, " W=")))

# Reshape the metric columns so each row is one allocation cap and each column
# is one model type.
sharpe_mat = reshape(df.Sharpe, 2, :)'
return_mat = reshape(df.TotalReturn_pct, 2, :)'
mad_mat = reshape(df.RealizedMAD_pct, 2, :)'

# ============================================================================
# Plot Sharpe ratio for each allocation cap.
# ============================================================================
p1 = bar(
    caps,
    sharpe_mat;
    label = ["Rolling" "Expanding"],
    title = "Allocation-cap sensitivity: Sharpe ratio",
    xlabel = "Allocation cap",
    ylabel = "Sharpe ratio",
    xrotation = 35
)
display(p1)

# ============================================================================
# Plot total return for each allocation cap.
# ============================================================================
p2 = bar(
    caps,
    return_mat;
    label = ["Rolling" "Expanding"],
    title = "Allocation-cap sensitivity: total return",
    xlabel = "Allocation cap",
    ylabel = "Total return (%)",
    xrotation = 35
)
display(p2)

# ============================================================================
# Plot realized MAD for each allocation cap.
# ============================================================================
p3 = bar(
    caps,
    mad_mat;
    label = ["Rolling" "Expanding"],
    title = "Allocation-cap sensitivity: realized MAD",
    xlabel = "Allocation cap",
    ylabel = "Realized MAD (%)",
    xrotation = 35
)
display(p3)
