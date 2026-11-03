import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mpl = plt.matplotlib 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from astropy.time import Time
from barycorrpy import get_BC_vel
import matplotlib.lines as mlines

grass_on_SSD_4parameter = h5py.File("../../RM/CB/data/neid_all_lines_rv_on_SSD_4parameter_gpu_ext.jld2", "r")
grass_on_RV_SSD_4parameter = grass_on_SSD_4parameter["rv_asymmetric_lsf"][()]

disco_SSD_4parameter = h5py.File("data/neid_all_lines_rv_disco_original.jld2", "r")
disco_RV_SSD_4parameter = disco_SSD_4parameter["rv_asymmetric_lsf"][()]

data = pd.read_csv("../../../data/NEID_Data.csv")
UTC_time = []
time_julian = []
for i in data["obsdate"][15:-150]:
    dt = datetime.strptime(i, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=27.5)
    UTC_time.append(dt)
    time_julian.append((Time(dt)).jd)

line_data = h5py.File("../../../data/neid_RVlinebyline.jld2", "r")
lines = line_data["name"][()]
line_rv  = line_data["rv"][()]

vb, warnings, flag = get_BC_vel(JDUTC=time_julian[0:-28], lat=31.9583 , longi=-111.5967, alt=209.7938, SolSystemTarget='Sun', predictive=False,zmeas=0.0)
UTC_time = UTC_time[0:-28]

def jld2_read(jld2_file, variable, vb, index):
    array = jld2_file[variable[index]][()]
    array = np.array(array + vb)
    array -= np.nanmean(array)
    return array

line_list = ["FeI_5250.2","FeI_5250.5","FeI_5379","TiII_5381","FeI_5382","FeI_5383","MnI_5432","FeI_5432","FeI_5434","NiI_5435","FeI_5436.3","FeI_5436.6","FeI_5576","NiI_5578","FeII_6149","FeI_6151","CaI_6169.0","CaI_6169.5","FeI_6170","FeI_6173","FeI_6301","FeI_6302"]

def bin_array(arr, bin_size):
    arr = np.array(arr)
    n_bins = len(arr) // bin_size
    return np.array([np.nanmean(arr[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])

line_indices = [0, 15, 19]

# Create a 2-row x 3-column grid
# hspace=0 removes vertical space between top RV plot and bottom Residuals plot
fig, axs = plt.subplots(
    2, 3, 
    figsize=(18, 7), 
    sharex=True, 
    sharey='row', 
    gridspec_kw={'hspace': 0, 'wspace': 0.18, 'height_ratios': [3, 1]}
)

for col_idx, i in enumerate(line_indices):
    ax_top = axs[0, col_idx]
    ax_bot = axs[1, col_idx]
    
    # Read data
    line_rv_array = jld2_read(line_data, line_rv, vb, i)
    grass_on_SSD_4parameter_array = jld2_read(grass_on_SSD_4parameter, grass_on_RV_SSD_4parameter, vb, i)
    disco_SSD_4parameter_array = jld2_read(disco_SSD_4parameter, disco_RV_SSD_4parameter, vb, i)

    # Compute RMS
    rms2 = round(np.sqrt((np.nansum((bin_array(line_rv_array, 4) - bin_array(disco_SSD_4parameter_array, 4))**2)) / len(bin_array(line_rv_array, 4))), 2)
    rms3 = round(np.sqrt((np.nansum((bin_array(line_rv_array, 4) - bin_array(grass_on_SSD_4parameter_array, 4))**2)) / len(bin_array(line_rv_array, 4))), 2)

    # ---------------- TOP PANEL: RVs ----------------
    ax_top.scatter(UTC_time, line_rv_array, color='k', marker="x", s=25)
    ax_top.plot(UTC_time, disco_SSD_4parameter_array, color='y', linestyle="--", linewidth=2, label=f"Model DISCO: {rms2} m/s")
    ax_top.plot(UTC_time, grass_on_SSD_4parameter_array, color='r', linestyle="--", linewidth=2, label=f"Model GRASS: {rms3} m/s")

    # Column Title (Spectral Line Name/Wavelength)
    ax_top.set_title(f"{line_list[i]} $\\AA$", fontsize=12, pad=8)
    ax_top.legend(fontsize=12, frameon=False, loc="lower right")
    ax_top.tick_params(axis='y', labelsize=12)

    # ---------------- BOTTOM PANEL: Residuals ----------------
    ax_bot.scatter(UTC_time, line_rv_array - disco_SSD_4parameter_array, color='y', marker="x", s=8)
    ax_bot.scatter(UTC_time, line_rv_array - grass_on_SSD_4parameter_array, color='r', marker="x", s=8)

    # X-axis time formatting on bottom row
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_bot.set_xlabel("Time (UTC)", fontsize=12)
    ax_bot.tick_params(axis='x', labelsize=12, rotation=30)
    ax_bot.tick_params(axis='y', labelsize=12)

    # Only show Y-axis labels on the leftmost column to keep layout clean
    if col_idx == 0:
        ax_top.set_ylabel("RV (m/s)", fontsize=12)
        ax_bot.set_ylabel("Residuals (m/s)", fontsize=12)

# Adjust layout and save combined 3-column figure
plt.tight_layout()
plt.savefig("disco_v_grass_rm.png", dpi=300, bbox_inches='tight')
plt.show()
