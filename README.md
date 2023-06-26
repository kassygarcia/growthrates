# growthrates
import numpy as np
import pandas as pd
import string
import os
import glob
import csv
import matplotlib
import matplotlib.pyplot as plt
import git
TABLES = ["OD600"]
workdir = os.getcwd().split('/')[-1]
date = '20230207'
run_no = '1'
plate = '1'
folder = glob.glob(date + '_r' + run_no + '_plate' + plate + '/*')
filename = '20230207_r1_plate1/20230207_plate_layout.xlsx'
df=pd.read_excel(filename)
df
xl = pd.ExcelFile(filename)
layout = xl.sheet_names
layout_info=list()
for l in layout:
    info=pd.read_excel(filename, sheet_name=l, header=None).values
    info=info.ravel()
    layout_info.append(info)

data=pd.read_csv('20230207_r1_plate1/20230207_r1_plate1_n.txt', header=None)
print(data)
columns=['time_min', 'temp_C', 'OD600'] + layout
df=pd.DataFrame(columns=columns)
for i, col in enumerate(data.loc[:,2:].columns):
    df_well=pd.DataFrame(columns=columns)
    df_well['time_min']=data.loc[:, 0]
    df_well['temp_C']=data.loc[:, 1]
    df_well['OD600']=data[col]
    for l, param in enumerate(layout):
        df_well[param]=layout_info[l][i]
    df=pd.concat([df, df_well])
blank_mean= df.loc[df['strain'] == "blank", :].groupby('time_min')['OD600'].apply(np.mean).reset_index()
df['OD600_norm']=0
for time in np.unique(df['time_min'].values):
    df.loc[df['time_min']==time,'OD600_norm']=df.loc[df['time_min']==time,'OD600'] - blank_mean.loc[blank_mean['time_min']==time, 'OD600'].values[0]
    df=df.loc[df['strain']!='blank',:]
    df=df.loc[df['strain']!='n',:]
blank_mean.loc[blank_mean['time_min']=='0007','OD600'].values
df['nl_OD600']=np.log10(df['OD600_norm'])
df1 = df[df['time_min'] > '0400']
df2 = df1[df1['time_min'] < '0520']
df2
df0=df2.iloc[680:697]
df0
MG1655_1_plot_1 = df0.loc[df0['strain']=="MG1655_1",:].groupby('time_min')['nl_OD600'].apply(np.mean).reset_index()
MG1655_1_plot_1
x="time_min"
y="nl_OD600"
MG1655_1_plot_1.plot(x, y)
time=MG1655_1_plot_1['time_min']
growth=MG1655_1_plot_1['nl_OD600']
time = time.apply(lambda x: float(x))
plt.scatter(time,growth)
plt.xlabel('time(min)')
plt.ylabel('nl_OD600')
plt.show
from scipy.optimize import curve_fit
def test_func(x,A,B):
    return A*x+B

parameters, variance=curve_fit(test_func, time, growth)
slope = parameters [0]
slope
