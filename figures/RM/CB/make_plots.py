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

#these projected models are for N=197, sub=40x40, with extinction, with sunspots 
    # with CB optimized polynomial from projected RVs (model 2 has GRASS CB)
    # (flux & CB effect), no MF
projected_SSD_4parameter = h5py.File("data/projected_SSD_4parameter_gpu_ext_CB_optim.jld2", "r")
projected_RV_SSD_4parameter = projected_SSD_4parameter["RV_list_no_cb"][()]

grass_on_SSD_4parameter = h5py.File("data/neid_all_lines_rv_on_SSD_4parameter_gpu_ext.jld2", "r")
grass_on_RV_SSD_4parameter = grass_on_SSD_4parameter["rv_asymmetric_lsf"][()]

grass_off_SSD_4parameter = h5py.File("data/neid_all_lines_rv_off_SSD_4parameter_gpu_ext_CB_optim.jld2", "r")
grass_off_RV_SSD_4parameter = grass_off_SSD_4parameter["rv_asymmetric_lsf"][()]

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
    array -= array[-1]
    return array

line_list = ["FeI_5250.2","FeI_5250.5","FeI_5379","TiII_5381","FeI_5382","FeI_5383","MnI_5432","FeI_5432","FeI_5434","NiI_5435","FeI_5436.3","FeI_5436.6","FeI_5576","NiI_5578","FeII_6149","FeI_6151","CaI_6169.0","CaI_6169.5","FeI_6170","FeI_6173","FeI_6301","FeI_6302"]
avg_photon_noise = [1.8211758584280178, 1.95054883079038, 2.144446659177623, 2.1658907566938175, 2.004090320949803, 2.596217972425998, 2.1135679819576634, 2.1978020386922146, 2.7521158957968668, 2.1516080460065217, 2.1281340447634953, 2.156445219458067, 3.029994032965276, 2.688447833232973, 3.1333750313001123, 3.341592265251292, 3.9849485492751655, 4.2694904053037765, 3.9731973397147713, 3.6154228169078944, 4.429243144558249, 4.150029318259546]

def bin_array(arr, bin_size):
    arr = np.array(arr)
    n_bins = len(arr) // bin_size
    return np.array([np.nanmean(arr[i*bin_size:(i+1)*bin_size]) for i in range(n_bins)])

rms1_arr = []
rms2_arr = []
rms3_arr = []
rms1_combined = []

for i in range(0,len(line_list)):    
    line_rv_array = jld2_read(line_data, line_rv, vb, i)
    projected_RV_SSD_4parameter_array = jld2_read(projected_SSD_4parameter, projected_RV_SSD_4parameter, vb, i)
    grass_on_SSD_4parameter_array = jld2_read(grass_on_SSD_4parameter, grass_on_RV_SSD_4parameter, vb, i)
    grass_off_SSD_4parameter_array = jld2_read(grass_off_SSD_4parameter, grass_off_RV_SSD_4parameter, vb, i)

    fig, axs = plt.subplots(2, figsize=(7,8), sharex=True, sharey=False, gridspec_kw={'hspace': 0, 'height_ratios': [3, 1]})
    axs[0].scatter(UTC_time, line_rv_array, color = 'k', marker = "x", s = 25)#, label = "NEID {} $\AA$ RVs".format(line_list[i])) 
    
    rms1 = (round(np.sqrt((np.nansum((bin_array(line_rv_array,4) - bin_array(projected_RV_SSD_4parameter_array,4))**2))/len(bin_array(line_rv_array,4))),2))
    rms2 = (round(np.sqrt((np.nansum((bin_array(line_rv_array,4) - bin_array(grass_off_SSD_4parameter_array,4))**2))/len(bin_array(line_rv_array,4))),2))
    rms3 = (round(np.sqrt((np.nansum((bin_array(line_rv_array,4) - bin_array(grass_on_SSD_4parameter_array,4))**2))/len(bin_array(line_rv_array,4))),2))
    rms1_arr.append(rms1)
    rms2_arr.append(rms2)
    rms3_arr.append(rms3)
    if rms1 <10: 
        rms1_combined.append(rms1)

    axs[0].plot(UTC_time, projected_RV_SSD_4parameter_array, color = 'b', linestyle = "--", linewidth = 2, label = "Model I-CB: {} m/s".format(rms1))
    axs[0].plot(UTC_time, grass_off_SSD_4parameter_array, color = 'y', linestyle = "--", linewidth = 2, label = "Model II-CB: {} m/s".format(rms2))
    axs[0].plot(UTC_time, grass_on_SSD_4parameter_array, color = 'r', linestyle = "--", linewidth = 2, label = "Model III: {} m/s".format(rms3))

    axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[0].set_xlabel("Time (UTC)", fontsize=12)
    axs[0].set_ylabel("RV (m/s)", fontsize=12)
    axs[0].legend(fontsize=12, frameon=False, loc="lower right")
    axs[0].tick_params(axis='y', labelsize=12)

    # residuals 
    axs[1].scatter(UTC_time, line_rv_array - projected_RV_SSD_4parameter_array, color = 'b', marker = "x", s = 8) 
    axs[1].scatter(UTC_time, line_rv_array - grass_off_SSD_4parameter_array, color = 'y', marker = "x", s = 8)  
    axs[1].scatter(UTC_time, line_rv_array - grass_on_SSD_4parameter_array, color = 'r', marker = "x", s = 8) 
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axs[1].set_xlabel("Time (UTC)", fontsize=12)
    axs[1].set_ylabel("Residuals", fontsize=12) 
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # axs[0].set_title("Fe I 5434 Å", fontsize=12)
    plt.tick_params(axis='x', which='both', top=True, labeltop=False)
    plt.savefig("{}.png".format(line_list[i]))
    # plt.clf()

plt.figure(figsize=(12, 6))
plt.scatter(lines, rms1_arr, color= 'b', label = "Model I-CB",s= 50)
plt.scatter(lines, rms2_arr, color= 'y', label = "Model II-CB-MF",s=50)
plt.scatter(lines, rms3_arr,color= 'r', label = 'Model III',s=50)
plt.scatter(lines, avg_photon_noise, color = 'k', marker='x', label = "Photon Noise")
plt.ylabel("RV RMS (m/s)", fontsize=12)
plt.xticks(rotation=60)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig("line_comps.pdf", bbox_inches='tight')
plt.clf() 

print(len(rms1_combined)/np.sum(1/np.array(rms1_combined)))
