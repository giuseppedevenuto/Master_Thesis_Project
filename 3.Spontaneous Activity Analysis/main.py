from NSA2                   import *
from tkinter.simpledialog   import askinteger
import pandas as pd
import xlsxwriter

rec_time = askinteger(" ", "Recording time (in seconds):") # in seconds
cc_type = 'binary signal'               # <EDITABLE> Method to calculate the CC: "point process" "binary signal"
cc_bin_time = 0.2                      # <EDITABLE> Bin time for Pearson's Coefficient [s]

FIRST_CONF = 1043       # <EDITABLE>
LAST_CONF = 1043        # <EDITABLE>
CONFIGS = np.arange(FIRST_CONF-1,LAST_CONF)+1

pc_config = []
isi_median_config = []
ibi_median_config = []
bl_median_config = []
bi_config = []
isi_config = []
mfr_config = []
ibi_config = []
mbr_config = []
bl_config = []
cv_mean_config = []
cv_std_config = []
rmse_config = []
for iconfig in CONFIGS:
    pc, isi_median, ibi_median, bl_median, bi, isi, mfr, ibi, mbr, bl, cv_mean, cv_std, rmse = NSA("CONFIG"+str(iconfig)+"-" , 'RFA', rec_time, cc_type, cc_bin_time)
    pc_config.append(pc[0])
    isi_median_config.append(isi_median[0])
    ibi_median_config.append(ibi_median[0])
    bl_median_config.append(bl_median[0])
    bi_config.append(bi[0])
    isi_config.append(isi[0])
    mfr_config.append(mfr[0])
    ibi_config.append(ibi[0])
    mbr_config.append(mbr[0])
    bl_config.append(bl[0])
    cv_mean_config.append(cv_mean[0])
    cv_std_config.append(cv_std[0])
    rmse_config.append(rmse[0])
# ext = ['svg','jpg']
# savedir = './Thesis Images' # <EDIT>
# fname = "LogConfigsComparison"
# scale = 'log'
# plot_configs([*pc_config, pc[1]], [*isi_median_config, isi_median[1]], [*ibi_median_config, ibi_median[1]], [*bl_median_config, bl_median[1]], [*bi_config, bi[1]], [*isi_config, isi[1]], [*mfr_config, mfr[1]], [*ibi_config, ibi[1]], [*mbr_config, mbr[1]], [*bl_config, bl[1]], [*cv_mean_config, cv_mean[1]], [*cv_std_config, cv_std[1]], configsName = CONFIGS, scale = scale, path = savedir, ext = ext, fname = fname)
# fname = "LinConfigsComparison"
# scale = 'lin'
# plot_configs([*pc_config, pc[1]], [*isi_median_config, isi_median[1]], [*ibi_median_config, ibi_median[1]], [*bl_median_config, bl_median[1]], [*bi_config, bi[1]], [*isi_config, isi[1]], [*mfr_config, mfr[1]], [*ibi_config, ibi[1]], [*mbr_config, mbr[1]], [*bl_config, bl[1]], [*cv_mean_config, cv_mean[1]], [*cv_std_config, cv_std[1]], configsName = CONFIGS, scale = scale, path = savedir, ext = ext, fname = fname)

idx = [0, *CONFIGS]
col = ['PC median', 'PC 25perc', 'PC 75perc', 
       'MFR median', 'MFR 25perc', 'MFR 75perc', 
       'ISI median', 'ISI 25perc', 'ISI 75perc', 
       'MBR median', 'MBR 25perc', 'MBR 75perc', 
       'IBI median', 'IBI 25perc', 'IBI 75perc',
       'BL median', 'BL 25perc', 'BL 75perc', 
       'BI median', 'BI 25perc', 'BI 75perc',
       'RMSE median', 'RMSE 25perc', 'RMSE 75perc']
table = []

iconf = []

iconf.append(np.median(pc[1]))
iconf.append(np.percentile(pc[1], 25))
iconf.append(np.percentile(pc[1], 75))

iconf.append(np.median(mfr[1]))
iconf.append(np.percentile(mfr[1], 25))
iconf.append(np.percentile(mfr[1], 75))

iconf.append(np.median(isi_median[1]))
iconf.append(np.percentile(isi_median[1], 25))
iconf.append(np.percentile(isi_median[1], 75))

iconf.append(np.median(mbr[1]))
iconf.append(np.percentile(mbr[1], 25))
iconf.append(np.percentile(mbr[1], 75))

iconf.append(np.median(ibi_median[1]))
iconf.append(np.percentile(ibi_median[1], 25))
iconf.append(np.percentile(ibi_median[1], 75))

iconf.append(np.median(bl_median[1]))
iconf.append(np.percentile(bl_median[1], 25))
iconf.append(np.percentile(bl_median[1], 75))

iconf.append(np.median(bi[1]))
iconf.append(np.percentile(bi[1], 25))
iconf.append(np.percentile(bi[1], 75))

iconf.append(np.median(rmse[1]))
iconf.append(np.percentile(rmse[1], 25))
iconf.append(np.percentile(rmse[1], 75))

table.append(iconf)

for i in range(len(CONFIGS)):
    iconf = []

    pc_median = np.median(pc_config[i])
    iconf.append(pc_median)
    if np.isnan(pc_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(pc_config[i], 25))
        iconf.append(np.percentile(pc_config[i], 75))

    mfr_median = np.median(mfr_config[i])
    iconf.append(mfr_median)
    if np.isnan(mfr_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(mfr_config[i], 25))
        iconf.append(np.percentile(mfr_config[i], 75))

    isi_median_median = np.median(isi_median_config[i])
    iconf.append(isi_median_median)
    if np.isnan(isi_median_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(isi_median_config[i], 25))
        iconf.append(np.percentile(isi_median_config[i], 75))

    mbr_median = np.median(mbr_config[i])
    iconf.append(mbr_median)
    if np.isnan(mbr_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(mbr_config[i], 25))
        iconf.append(np.percentile(mbr_config[i], 75))

    ibi_median_median = np.median(ibi_median_config[i])
    iconf.append(ibi_median_median)
    if np.isnan(ibi_median_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(ibi_median_config[i], 25))
        iconf.append(np.percentile(ibi_median_config[i], 75))

    bl_median_median = np.median(bl_median_config[i])
    iconf.append(bl_median_median)
    if np.isnan(bl_median_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(bl_median_config[i], 25))
        iconf.append(np.percentile(bl_median_config[i], 75))

    bi_median = np.median(bi_config[i])
    iconf.append(bi_median)
    if np.isnan(bi_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(bi_config[i], 25))
        iconf.append(np.percentile(bi_config[i], 75))

    rmse_median = np.median(rmse_config[i])
    iconf.append(rmse_median)
    if np.isnan(rmse_median):
        iconf.append(float('NaN'))
        iconf.append(float('NaN'))
    else:
        iconf.append(np.percentile(rmse_config[i], 25))
        iconf.append(np.percentile(rmse_config[i], 75))

    table.append(iconf)

workbook = xlsxwriter.Workbook('Results1.xlsx') # <EDIT>
worksheet = workbook.add_worksheet()
workbook.close()

results = pd.DataFrame(table, index = idx, columns=col)
results.to_excel("Results1.xlsx", sheet_name="Sheet1") # <EDIT>