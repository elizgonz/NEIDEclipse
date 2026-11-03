using CSV
using CUDA
using GRASS
using Printf
using Revise
using PyCall
using DataFrames
using Statistics
using EchelleCCFs
using Polynomials
using BenchmarkTools
using HypothesisTests

# Plotting setup
import PyPlot; plt = PyPlot; mpl = plt.matplotlib; plt.ioff()

# Define 3 lines to plot across columns
lines_to_do = ["FeI_5250.2", "FeI_6151", "FeI_6173"]
lines_to_do_disco = ["Fe5250", "Fe6152", "Fe6173"]

# Polynomial parameters per line
poly_params = Dict(
    "FeI_5250.2" => (a = 1.2240059582206162, b = -1.768954606693554), #CB1 Error: ±0.005828189303758756, CB2 Error: ±0.008398371720637328
    "FeI_6151"   => (a = 0.48715649159711116, b = -0.8183702374327807), #CB1 Error: ±0.0020323983592821586, CB2 Error: ±0.0022623092429288733
    "FeI_6173"   => (a = 1.0732792946482501, b = -1.6328472192824552) #CB1 Error: ±0.004093005811283664, CB2 Error: ±0.006211424210429673
)

poly_params_disco = Dict(
    "FeI_5250.2" => (a = 2.0335737688143194, b = -2.811552452160309), #CB1 Error: ±0.003783873168339443, CB2 Error: ±0.0053184557898415296
    "FeI_6151"   => (a = 0.8784110049319205, b = -1.3993729728363453), #CB1 Error: ±0.0016746156648692941, CB2 Error: ±0.001650441427531251
    "FeI_6173"   => (a = 1.7691729927345508, b = -2.5553545981192034) #CB1 Error: ±0.0026317113206594175, CB2 Error: ±0.003920730354628624
)

# Markers and colors for observation directions
cs = ["k", "tab:blue", "tab:orange", "tab:green", "tab:pink"]
ms = [".", ".", ".", ".", "."]
lbls = ["Center", "North", "South", "East", "West"]

# Load GRASS Line Properties
lp = GRASS.LineProperties(exclude=["CI_5380", "NaI_5896"])
files = GRASS.get_file(lp)
line_names = GRASS.get_name(lp)

# Create a 2-row x 3-column figure grid
# sharex=true aligns mu across rows; sharey='row' aligns Y-scale across columns
fig, axs = plt.subplots(
    2, 3, 
    figsize=(16, 8), 
    sharex=true, 
    sharey="row",
    gridspec_kw=Dict("hspace" => 0.08, "wspace" => 0.08)
)

println("=================================================================")
println(" Generating 2-Row x 3-Column CB Comparison Plot")
println(" Top Row: Directional CB | Bottom Row: Axis-Averaged CB")
println("=================================================================")

for (col_idx, target_line) in enumerate(lines_to_do)
    ax_top = axs[1, col_idx] # Row 1: Directional CB
    ax_bot = axs[2, col_idx] # Row 2: Axis-Averaged CB
    
    # Find matching file in GRASS LineProperties
    file_idx = findfirst(name -> name == target_line || replace(name, "." => "") == replace(target_line, "." => ""), line_names)
    if isnothing(file_idx)
        @warn "Line $target_line not found in GRASS LineProperties, skipping..."
        continue
    end

    fname = files[file_idx]
    raw_line_name = line_names[file_idx]
    println("Column $col_idx: Processing $raw_line_name ($fname)...")

    # Read SolarData from GRASS
    soldata = GRASS.SolarData(fname=fname)

    # 1. Read corresponding DISCO CSV file
    csv_filename = "data/disco_$(lines_to_do_disco[col_idx])_cb.csv"
    
    if !isfile(csv_filename)
        possible_csvs = filter(f -> endswith(f, "_cb.csv") && contains(f, lines_to_do_disco[col_idx]), readdir("."))
        if !isempty(possible_csvs)
            csv_filename = possible_csvs[1]
        end
    end

    df_loaded = false
    local DISCOCB, PolynomialCB, MeasuredCB, mus_disco
    if isfile(csv_filename)
        println("  Found CSV: $csv_filename")
        df = CSV.read(csv_filename, DataFrame)
        DISCOCB = df.DISCOCB
        PolynomialCB = df.PolynomialCB
        MeasuredCB = df.MeasuredCB
        mus_disco = df.Mu
        df_loaded = true
    else
        @warn "  CSV file $csv_filename not found! DISCO curves will be omitted for $raw_line_name."
    end

    # 2. Extract GRASS SolarData points & setup mu averaging arrays
    mus = unique(soldata.mu)
    avgs = zeros(length(mus))
    cnts = zeros(length(mus))
    mus_plot = zeros(length(mus))
    seen = falses(length(ms))

    for (j, k) in enumerate(keys(soldata.cbs))
        ax_idx = GRASS.parse_ax_string(string(k[1]))
        mu_val = GRASS.parse_mu_string(string(k[2]))
        
        cbs_abs = soldata.cbs[k] * GRASS.c_ms

        # Accumulate sums and counts per mu
        mu_idx = findfirst(mu_val .== mus)
        if !isnothing(mu_idx)
            avgs[mu_idx] += cbs_abs
            cnts[mu_idx] += 1
            mus_plot[mu_idx] = mu_val
        end

        label_str = !seen[ax_idx + 1] ? lbls[ax_idx + 1] : nothing
        if !seen[ax_idx + 1]
            seen[ax_idx + 1] = true
        end

        # Top Panel: Directional CB (Each Axis)
        ax_top.errorbar(
            [mu_val], [cbs_abs];
            yerr=[5.0], fmt=ms[ax_idx + 1], color=cs[ax_idx + 1], capsize=3, label=label_str
        )
    end

    # Compute axis-averaged CB per mu
    valid_mask = cnts .> 0
    mus_sorted = mus_plot[valid_mask]
    mean_cbs = avgs[valid_mask] ./ cnts[valid_mask]

    # Sort by mu for clean plotting
    sort_idx = sortperm(mus_sorted)
    mus_sorted = mus_sorted[sort_idx]
    mean_cbs = mean_cbs[sort_idx]

    # Bottom Panel: Axis-Averaged Observations
    ax_bot.errorbar(
        mus_sorted, mean_cbs .- mean(mean_cbs);
        yerr=fill(5.0, length(mean_cbs)), fmt="o", color="k", ms=4, capsize=3, label="Rel. LARS Average"
    )

    fit_params = get(poly_params, target_line, (a = 1.4098, b = -2.0651))
    xfit = range(0, 1, length=500)
    yfit = fit_params.a .* xfit.^2 .+ fit_params.b .* xfit
    yfit_ms = yfit .* 1000.0

    ax_bot.plot(xfit, yfit_ms .- mean(yfit_ms), color="g", lw=2, label="Model II Optim CB")

    fit_params = get(poly_params_disco, target_line, (a = 1.4098, b = -2.0651))
    yfit = fit_params.a .* xfit.^2 .+ fit_params.b .* xfit
    yfit_ms = yfit .* 1000.0

    ax_bot.plot(xfit, yfit_ms .- mean(yfit_ms), color="b", lw=2, label="Model DISCO Optim CB")

    # 3. DISCO & Measured CB curves if CSV loaded
    if df_loaded
        # Top Panel: DISCO CB curve
        ax_top.plot(mus_disco, DISCOCB, color="pink", lw=2, label="Abs. DISCO CB")

        # Bottom Panel: DISCO CB curve
        ax_bot.plot(mus_disco, DISCOCB .- mean(filter(!isnan, DISCOCB)), color="pink", lw=2, label="Rel. DISCO CB")
    end

    # ---------------- FORMATTING ----------------
    # Top Panel Title
    title_str = replace(raw_line_name, "_" => " ")
    idx = findfirst('I', title_str)
    line_title = isnothing(idx) ? title_str : title_str[1:idx-1] * " " * title_str[idx:end] * " Å"
    ax_top.set_title(line_title, fontsize=12, pad=8)

    # Grid & Limits
    for ax in [ax_top, ax_bot]
        ax.grid(true, alpha=0.3)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xlim(1.02, 0.0) # Inverted mu axis (center 1.0 on left -> limb 0.0 on right)
    end

    # Bottom Row X-label
    ax_bot.set_xlabel("μ", fontsize=12)

    # Y-axis Labels & Legends on Leftmost Column
    if col_idx == 1
        ax_top.set_ylabel("Abs. Convective Blueshift (m/s)", fontsize=12)
        ax_bot.set_ylabel("Rel. Convective Blueshift (m/s)", fontsize=12)
        
        # Legends
        ax_top.legend(loc="upper left", fontsize=12, frameon=true, framealpha=0.8, ncols=2)
        ax_bot.legend(loc="upper left", fontsize=12, frameon=true, framealpha=0.8)
    end
end

plt.tight_layout()
out_fig = "cbs.png"
plt.savefig(out_fig, dpi=600, bbox_inches="tight")
println("\n=================================================================")
println(" Plot saved successfully to: ", out_fig)
println("=================================================================")