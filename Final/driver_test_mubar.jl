# ============================================================================
# Sensitivity test for target return mu_bar.
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

# Define the target return grid.
# MU_STEP is the monthly return increment.
# MU_MAX is the largest monthly target return tested.
MU_STEP = 0.001
MU_MAX = 0.050
MU_VALUES = collect(0.0:MU_STEP:MU_MAX)

# Store one performance row for each feasible model and target value.
rows = NamedTuple[]

println("\nTesting target return mu_bar")
println("="^90)

# Test all target return values in the grid.
for mu_bar in MU_VALUES
    println("\nTesting mu_bar = $(round(100 * mu_bar, digits = 2))% per month")

    # Track whether at least one model is feasible for the current target.
    any_feasible = false

     
    # Test the non-adaptive model.
     
    feasible_na, ret_na, W_na, cost_na =
        try_non_adaptive(R_train, R_test, mu_bar, asset_names)

    # If feasible, store the resulting performance metrics.
    if feasible_na
        any_feasible = true
        push!(
            rows,
            performance_row(
                "Non-adaptive μ=$(round(100 * mu_bar, digits = 2))%",
                ret_na,
                W_na,
                cost_na
            )
        )
    end

     
    # Test the rolling model.
     
    feasible_rl, ret_rl, W_rl, cost_rl =
        try_rolling(
            R_train,
            R_test,
            x_start,
            mu_bar,
            c_vec,
            DEFAULT_W_MAX,
            DEFAULT_USE_CAP,
            DEFAULT_L
        )

    # If feasible, store the resulting performance metrics.
    if feasible_rl
        any_feasible = true
        push!(
            rows,
            performance_row(
                "Rolling μ=$(round(100 * mu_bar, digits = 2))%",
                ret_rl,
                W_rl,
                cost_rl
            )
        )
    end

     
    # Test the expanding model with exponential forgetting.
     
    feasible_ex, ret_ex, W_ex, cost_ex =
        try_expanding(
            R_train,
            R_test,
            x_start,
            mu_bar,
            c_vec,
            DEFAULT_W_MAX,
            DEFAULT_USE_CAP,
            DEFAULT_LAMBDA
        )

    # If feasible, store the resulting performance metrics.
    if feasible_ex
        any_feasible = true
        push!(
            rows,
            performance_row(
                "Expanding μ=$(round(100 * mu_bar, digits = 2))%",
                ret_ex,
                W_ex,
                cost_ex
            )
        )
    end

    # Stop the search once all three models are infeasible for the current
    # target return. Higher targets are expected to be at least as difficult.
    if !any_feasible
        println("No model feasible at mu_bar = $(round(100 * mu_bar, digits = 2))%. Stopping search.")
        break
    end
end

# Convert collected results to a DataFrame for printing and plotting.
df = DataFrame(rows)

println("\nTarget-return sensitivity")
show(df, allrows = true, allcols = true)
println()

# Apply the shared plotting style.
init_plot_style()

# Names used to separate the results from the three model types.
model_names = ["Non-adaptive", "Rolling", "Expanding"]

# Extract the target return percentage from the model label.
# Example: "Rolling μ=1.9%" becomes 1.9.
function extract_mu(model_string)
    m = match(r"μ=([0-9.]+)%", model_string)
    return m === nothing ? NaN : parse(Float64, m.captures[1])
end

# Add the extracted target return values as a numeric column.
df[!, :Mu_pct] = extract_mu.(df.Model)

 
# Plot Sharpe ratio as a function of the required monthly return.
 
p1 = plot(
    title = "Target-return sensitivity: Sharpe ratio",
    xlabel = "Required monthly return μ₀ (%)",
    ylabel = "Sharpe ratio",
    legend = :best
)

for model in model_names
    # Select only the rows belonging to the current model type.
    sub = filter(row -> startswith(row.Model, model), df)

    # Add the model curve to the plot.
    plot!(
        p1,
        sub.Mu_pct,
        sub.Sharpe;
        lw = 3,
        marker = :circle,
        label = model
    )
end

display(p1)

 
# Plot total return as a function of the required monthly return.
 
p2 = plot(
    title = "Target-return sensitivity: total return",
    xlabel = "Required monthly return μ₀ (%)",
    ylabel = "Total return (%)",
    legend = :best
)

for model in model_names
    # Select only the rows belonging to the current model type.
    sub = filter(row -> startswith(row.Model, model), df)

    # Add the model curve to the plot.
    plot!(
        p2,
        sub.Mu_pct,
        sub.TotalReturn_pct;
        lw = 3,
        marker = :circle,
        label = model
    )
end

display(p2)

 
# Plot realized MAD as a function of the required monthly return.
 
p3 = plot(
    title = "Target-return sensitivity: realized MAD",
    xlabel = "Required monthly return μ₀ (%)",
    ylabel = "Realized MAD (%)",
    legend = :best
)

for model in model_names
    # Select only the rows belonging to the current model type.
    sub = filter(row -> startswith(row.Model, model), df)

    # Add the model curve to the plot.
    plot!(
        p3,
        sub.Mu_pct,
        sub.RealizedMAD_pct;
        lw = 3,
        marker = :circle,
        label = model
    )
end

display(p3)
