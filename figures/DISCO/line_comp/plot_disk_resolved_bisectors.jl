# plot_lars_bisectors.jl
# Plot LARS Disk-Resolved Line Bisectors vs. DISCO Model across 3 Spectral Lines in GRASS
# Centered at 0.0 m/s at the line core to directly compare bisector shapes and C-shapes.
# Creates a 3-column plot with dual x-axes (Relative Velocity & Wavelength)
# Averaging bisectors across all available solar observation axes at each mu.
using PyCall
using GRASS
using PyPlot
using Statistics
using Printf

"""
    get_mu_for_key(soldata, k)

Helper function to extract the limb angle μ (Float64) for a given key `k`
(tuple of `(axis, position)` symbols, e.g., `(:e, :mu05)`), regardless of whether
`soldata.mu` is a `Dict` or a `Vector{Float64}`.
"""
function get_mu_for_key(soldata, k)
    if soldata.mu isa AbstractDict
        if haskey(soldata.mu, k)
            return Float64(soldata.mu[k])
        elseif haskey(soldata.mu, k[2])
            return Float64(soldata.mu[k[2]])
        end
    elseif soldata.mu isa AbstractVector
        sym_str = string(k[2]) # e.g. "mu05"
        for m in soldata.mu
            m_val = Float64(m)
            m_str = replace(@sprintf("%.2f", m_val), "." => "")
            m_short = replace(string(m_val), "." => "")
            
            if endswith(sym_str, m_short) || endswith(sym_str, m_str) || contains(sym_str, m_short)
                return m_val
            end
        end
        
        if startswith(sym_str, "mu")
            raw_num = sym_str[3:end]
            if raw_num in ("10", "100", "1")
                return 1.0
            elseif length(raw_num) == 2
                return parse(Float64, raw_num) / 100.0
            elseif length(raw_num) == 3
                return parse(Float64, raw_num) / 1000.0
            end
        end
    end
    error("Could not determine μ for key $k in soldata")
end

"""
    get_cbs_for_key(soldata, k, λ0; c=2.99792458e8)

Helper function to extract the convective blueshift velocity (in m/s) for key `k`.
Handles cases where `soldata.cbs` is stored as dimensionless z, wavelength offset in Å, or m/s.
"""
function get_cbs_for_key(soldata, k, λ0; c=2.99792458e8)
    if !hasproperty(soldata, :cbs) && !hasfield(typeof(soldata), :cbs)
        return 0.0
    end
    
    cbs_obj = soldata.cbs
    val = 0.0
    if cbs_obj isa AbstractDict
        if haskey(cbs_obj, k)
            val = Float64(cbs_obj[k])
        elseif haskey(cbs_obj, k[2])
            val = Float64(cbs_obj[k[2]])
        end
    elseif cbs_obj isa AbstractVector
        val = Float64(mean(cbs_obj))
    elseif cbs_obj isa Number
        val = Float64(cbs_obj)
    end
    
    if abs(val) < 1e-3  # Dimensionless z = v/c
        return val * c
    elseif abs(val) < 10.0  # Wavelength offset in Angstroms
        return c * (val / λ0)
    else  # Already in m/s
        return val
    end
end

"""
    plot_lars_disk_resolved_bisectors(line_indices=[1, 16, 20]; 
                                     c=2.99792458e8, 
                                     out_fig="lars_disk_resolved_bisectors.png")

Extracts LARS disk-resolved observations from GRASS's `SolarData` for 3 spectral lines.
Averages bisector profiles at each unique μ value across all available solar observation axes 
(e.g., North, South, East, West) and timesteps.
Centers all bisectors at 0.0 m/s at the line core (index 1) to isolate and compare line shapes.
Overlays DISCO model bisectors (dashed lines) computed via `GRASS.calc_bisector`.
Plots a 3-column figure with:
  - Y-axis: Normalized Intensity (I / Ic)
  - Primary (Bottom) X-axis: Relative Velocity (m/s) centered at line core
  - Secondary (Top) X-axis: Absolute Wavelength (Å)
  - Direct μ text labels in matching colors
"""
function plot_lars_disk_resolved_bisectors(disco_base::String,line_indices::Vector{Int}=[1, 16, 20]; 
                                            c=2.99792458e8, 
                                            out_fig="disk_resolved_bisectors.png")
    pyimport("sys")["path"].insert(0, joinpath(disco_base, "line_creation"))
    make_lines = pyimport("make_lines")                                      

    lp = GRASS.LineProperties()
    names = GRASS.get_name(lp)
    λrests = GRASS.get_rest_wavelength(lp)
    files = GRASS.get_file(lp)

    num_lines = length(line_indices)
    @assert num_lines == 3 "Function expects exactly 3 line indices for a 3-column plot."

    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=true)
    cmap = plt.get_cmap("viridis")

    println("=================================================================")
    println(" Extracting & Plotting LARS & DISCO Disk-Resolved Bisectors")
    println("=================================================================")

    for (col_idx, l_idx) in enumerate(line_indices)
        disco_line_name, h5_file = if l_idx == 1
            ("Fe5250", "disco_Fe5250_params.h5")
        elseif l_idx == 16
            ("Fe6152", "disco_Fe6152_params.h5")
        elseif l_idx == 20
            ("Fe6173", "disco_Fe6173_params.h5")
        else
            error("Unsupported line index $l_idx")
        end

        py_gen = make_lines.LineProfileGenerator(disco_line_name)
        h5_path = joinpath(disco_base, "line_creation/data/$(disco_line_name)/$(h5_file)")
        
        p_host = GRASS.Eclipse.DISCOParams(h5_path)
        
        # Convert wavelength to Float64 Vector in Angstroms
        wl_angstroms = Float64.(p_host.wavelength .* 10.0f0)
        
        ax = axs[col_idx]
        line_name = names[l_idx]
        λ0 = Float64(λrests[l_idx])
        lfile = files[l_idx]

        println("\nColumn $col_idx: Processing Line $l_idx ($line_name at $λ0 Å, DISCO: $disco_line_name)...")

        soldata = GRASS.SolarData(lfile)
        all_keys = collect(keys(soldata.bis))

        all_mus = Float64[get_mu_for_key(soldata, k) for k in all_keys]
        mu_rounded = round.(all_mus, digits=2)
        unique_mus = sort(unique(mu_rounded), rev=true)  # 1.0 (center) -> 0.2 (limb)

        println("  Found $(length(unique_mus)) unique μ positions: ", unique_mus)

        for (m_idx, mu_val) in enumerate(unique_mus)
            deg = Float64(rad2deg(acos(clamp(mu_val, 0.0, 1.0))))
            
            # --- A. Python DISCO reference (median profiles, quantiles = 0.5) ---
            py_res = py_gen.get_full_profile(deg, GT_quantile=0.5, ratio_quantile=0.5)
            py_prof = Float64.(py_res[1])
            py_prof ./= maximum(py_prof)
            
            # Compute bisector using GRASS.calc_bisector on Float64 vectors
            py_bis_x, py_bis_y = GRASS.calc_bisector(wl_angstroms, py_prof)
            
            # Convert DISCO bisector wavelength (Å) to Doppler velocity (m/s) relative to λ0
            py_bis_v = c .* ((py_bis_x .- λ0) ./ λ0)

            # --- B. LARS Observed Bisectors ---
            matching_keys = [all_keys[idx] for (idx, m) in enumerate(mu_rounded) if abs(m - mu_val) < 0.02]
            n_axes = length(matching_keys)

            bis_axis_avgs = [vec(mean(soldata.bis[k], dims=2)) for k in matching_keys]
            int_axis_avgs = [vec(mean(soldata.int[k], dims=2)) for k in matching_keys]

            avg_bis_offset = sum(bis_axis_avgs) ./ length(bis_axis_avgs)
            avg_int = sum(int_axis_avgs) ./ length(int_axis_avgs)

            # Convective blueshift velocity for this μ position
            cbs_m_s = mean([get_cbs_for_key(soldata, k, λ0; c=c) for k in matching_keys])

            # Total Doppler velocity (m/s) = shape offset + convective blueshift
            if maximum(abs.(avg_bis_offset)) < 50.0
                v_shape = c .* (avg_bis_offset ./ λ0)
            else
                v_shape = avg_bis_offset
            end
            
            v_total_m_s = v_shape .+ cbs_m_s

            # Center bisectors at zero velocity by subtracting their mean velocity
            v_total_m_s .-= mean(v_total_m_s)
            py_bis_v .-= mean(py_bis_v)

            color_norm = clamp((mu_val - 0.2) / (1.0 - 0.2), 0.0, 1.0)
            line_color = cmap(color_norm)

            # Plot LARS (Solid) and DISCO (Dashed)
            mu_label = @sprintf("μ = %.2f", mu_val)
            ax.plot(v_total_m_s, avg_int, color=line_color, linestyle="-", linewidth=1.8, label=mu_label)
            ax.plot(py_bis_v, py_bis_y, color=line_color, linestyle="--", linewidth=1.8)
        end

        # ---------------- AXIS FORMATTING ----------------
        ax.set_title(@sprintf("%s Å", line_name), fontsize=12, pad=28)
        ax.set_xlabel("Relative Velocity from Mean (m/s)", fontsize=12)
        ax.grid(true, alpha=0.3)
        ax.tick_params(axis="both", labelsize=12)

        if col_idx == 1
            ax.set_ylabel("Normalized Intensity", fontsize=12)
            # Style indicator legend for LARS vs DISCO
            style_handles = [
                PyPlot.matplotlib.lines.Line2D([], [], color="black", linestyle="-", linewidth=1.8),
                PyPlot.matplotlib.lines.Line2D([], [], color="black", linestyle="--", linewidth=1.8)
            ]
            # ax.legend(handles=style_handles, loc="lower left", fontsize=12, frameon=true, framealpha=0.85)
        elseif col_idx == 2
            # μ value legend
            ax.legend(fontsize=12, ncol=2, loc="lower right", frameon=true, framealpha=0.85)
        end

        # --- Secondary Top X-Axis for Wavelength (Å) ---
        secax = ax.secondary_xaxis("top", functions=(
            v -> λ0 .* (1.0 .+ v ./ c),          # Forward: Velocity -> Wavelength
            w -> c .* ((w .- λ0) ./ λ0)          # Inverse: Wavelength -> Velocity
        ))
        secax.set_xlabel("Wavelength (Å)", fontsize=12, labelpad=7)
        secax.ticklabel_format(useOffset=false, style="plain")
        secax.xaxis.set_major_formatter(PyPlot.matplotlib.ticker.FormatStrFormatter("%.2f"))
        secax.tick_params(labelsize=12, rotation=20)
    end

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    println("\n=================================================================")
    println(" Plot saved successfully to: ", out_fig)
    println("=================================================================")
end

disco_base = "../../../../../sw/DISCO"
plot_lars_disk_resolved_bisectors(disco_base, [1, 16, 20])