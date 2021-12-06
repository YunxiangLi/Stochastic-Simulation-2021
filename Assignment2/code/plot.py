# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 21:19:03 2021

@author: Yunxiang
"""

import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pandas as pd
from scipy import stats


df1 = pd.read_csv('rho1.csv')
df2 = pd.read_csv('rho2.csv')
df4 = pd.read_csv('rho4.csv')

# df = pd.concat([df1,df2,df4], ignore_index = True)
# del df['Unnamed: 0']


# mean = df.groupby(by = ['rho','number of servers','number of customers']).mean()['mean waiting time']
# std = df.groupby(by = ['rho','number of servers','number of customers']).std()['mean waiting time']

# mean.to_csv('mean.csv')
# std.to_csv('std.csv')

mean_df = pd.read_csv('mean.csv')
std_df = pd.read_csv('std.csv')


# compare across n & rho mean std
fig, axes = plt.subplots(nrows=3, ncols=2, sharex = True, figsize = (14,14))

sns.lineplot(ax=axes[0,1], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("magma", 4), data=std_df[std_df['number of servers']==1])
axes[0,1].legend(fontsize = 20)
axes[0,1].set_title('n = 1', fontsize = 25)
axes[0,1].set_ylabel('Standard deviation', fontsize = 20)

sns.lineplot(ax=axes[1,1], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("magma", 4), data=std_df[std_df['number of servers']==2])
axes[1,1].legend(fontsize = 20)
axes[1,1].set_title('n = 2', fontsize = 25)
axes[1,1].set_ylabel('Standard deviation', fontsize = 20)

sns.lineplot(ax=axes[2,1], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("magma", 4), data=std_df[std_df['number of servers']==4])
axes[2,1].legend(fontsize = 20)
axes[2,1].set_title('n = 4', fontsize = 25)
axes[2,1].set_ylabel('Standard deviation', fontsize = 20)
axes[2,1].set_xlabel('Number of measurements (customers)', fontsize = 20)

sns.lineplot(ax=axes[0,0], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("mako", 4), data=mean_df[mean_df['number of servers']==1])
axes[0,0].legend(fontsize = 20, loc = 'upper right')
axes[0,0].set_title('n = 1', fontsize = 25)
axes[0,0].set_ylabel('Mean', fontsize = 20)

sns.lineplot(ax=axes[1,0], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("mako", 4), data=mean_df[mean_df['number of servers']==2])
axes[1,0].legend(fontsize = 20, loc = 'upper right')
axes[1,0].set_title('n = 2', fontsize = 25)
axes[1,0].set_ylabel('Mean', fontsize = 20)

sns.lineplot(ax=axes[2,0], x = "number of customers", y = "mean waiting time", hue = "rho", palette = sns.color_palette("mako", 4), data=mean_df[mean_df['number of servers']==4])
axes[2,0].legend(fontsize = 20, loc = 'upper right')
axes[2,0].set_title('n = 4', fontsize = 25)
axes[2,0].set_ylabel('Mean', fontsize = 20)
axes[2,0].set_xlabel('Number of measurements (customers)', fontsize = 20)

fig.tight_layout()
fig.savefig('mean_std', dpi = 1000)

histplot
fig, axes = plt.subplots(nrows=3, ncols=2, figsize = (10,10))

wait_n1r7 = df[(df['rho'] == 0.7) & (df['number of servers'] == 1) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax = axes[0,0], data = wait_n1r7, kde = True, stat = 'probability')
p = stats.shapiro(wait_n1r7)[1]
axes[0,0].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4f}', fontsize = 11)
axes[0,0].xaxis.label.set_visible(False)

wait_n2r7 = df[(df['rho'] == 0.7) & (df['number of servers'] == 2) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax=axes[1,0], data = wait_n2r7, kde = True, stat = 'probability')
p = stats.shapiro(wait_n2r7)[1]
axes[1,0].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4f}', fontsize = 11)
axes[1,0].xaxis.label.set_visible(False)


wait_n4r7 = df[(df['rho'] == 0.7) & (df['number of servers'] == 4) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax=axes[2,0], data = wait_n4r7, kde = True, stat = 'probability')
p = stats.shapiro(wait_n4r7)[1]
axes[2,0].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4f}', fontsize = 11)


wait_n1r9 = df[(df['rho'] == 0.95) & (df['number of servers'] == 1) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax = axes[0,1], data = wait_n1r9, kde = True, stat = 'probability')
p = stats.shapiro(wait_n1r9)[1]
axes[0,1].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4E}', fontsize = 11)
axes[0,1].xaxis.label.set_visible(False)

wait_n2r9 = df[(df['rho'] == 0.95) & (df['number of servers'] == 2) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax=axes[1,1], data = wait_n2r9, kde = True, stat = 'probability')
p = stats.shapiro(wait_n2r9)[1]
axes[1,1].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4E}', fontsize = 11)
axes[1,1].xaxis.label.set_visible(False)


wait_n4r9 = df[(df['rho'] == 0.95) & (df['number of servers'] == 4) & (df['number of customers'] == 100000)]['mean waiting time']
sns.histplot(ax=axes[2,1], data = wait_n4r9, kde = True, stat = 'probability')
p = stats.shapiro(wait_n4r9)[1]
axes[2,1].set_title(r'n = 1, $\rho=0.7$  ' + f'p_value = {p:.4E}', fontsize = 11)

fig.savefig('normality', dpi = 500)


# M/M/1 Short Job Priority
df_sjp = pd.read_csv('sjp.csv')
df1['Queue type'] = 'First In, First Out'
df_sjp['Queue type'] = 'Short Job Priority'

fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (8,6), sharex = True)
sns.lineplot(ax = axes[0], x = "number of customers", y = "mean waiting time", color = 'xkcd:orange', data = df_sjp[df_sjp['number of customers'] >= 3000])
axes[0].set_ylabel('Average waiting time', fontsize = 15)
axes[0].set_title('Short Job Priority', color = 'xkcd:orange', fontsize = 15)
sns.lineplot(ax = axes[1], x = "number of customers", y = "mean waiting time", color = 'xkcd:blue', data = df1[(df1['number of customers'] >= 3000) & (df1['rho'] == 0.9)])
axes[1].set_xlabel('Number of measurements (customers)', fontsize = 15)
axes[1].set_ylabel('Average waiting time', fontsize = 15)
axes[1].set_title('First In, First Out', color = 'xkcd:blue', fontsize = 15)

fig.tight_layout()
fig.savefig('sjp', dpi = 1000)


# M/D/n = 1,2,4
md1 = pd.read_csv('md1.csv')
md2 = pd.read_csv('md2.csv')
md4 = pd.read_csv('md4.csv')


fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (8,6), sharex = True)
sns.lineplot(ax = axes[0], x = "number of customers", y = "mean waiting time", color = 'xkcd:green', data = md1[md1['number of customers'] >= 3000])
axes[0].set_ylabel('Average waiting time', fontsize = 10)
axes[0].set_title('M/D/1', color = 'xkcd:green', fontsize = 15)
sns.lineplot(ax = axes[1], x = "number of customers", y = "mean waiting time", color = 'xkcd:blue', data = df1[(df1['number of customers'] >= 3000) & (df1['rho'] == 0.9)])
axes[1].set_xlabel('Number of measurements (customers)', fontsize = 15)
axes[1].set_ylabel('Average waiting time', fontsize = 10)
axes[1].set_title('M/M/1', color = 'xkcd:blue', fontsize = 15)

fig.tight_layout()
fig.savefig('md1', dpi = 1000)


# M/H/1
mh1 = pd.read_csv('mh1.csv')
mm1_mu = pd.read_csv('mm1_mu.csv')

fig, axes = plt.subplots(nrows=2, ncols=1, figsize = (8,6), sharex = True)
sns.lineplot(ax = axes[0], x = "number of customers", y = "mean waiting time", color = 'xkcd:red', data = mh1[mh1['number of customers'] >= 1000])
axes[0].set_ylabel('Average waiting time', fontsize = 10)
axes[0].set_title(r'M/H/1, $\mu=0.5$', color = 'xkcd:red', fontsize = 15)
sns.lineplot(ax = axes[1], x = "number of customers", y = "mean waiting time", color = 'xkcd:blue', data = mm1_mu[mm1_mu['number of customers'] >= 1000])
axes[1].set_xlabel('Number of measurements (customers)', fontsize = 15)
axes[1].set_ylabel('Average waiting time', fontsize = 10)
axes[1].set_title(r'M/M/1, $\mu=0.5$', color = 'xkcd:blue', fontsize = 15)

fig.tight_layout()
fig.savefig('mh1', dpi = 1000)

# sjp2 = pd.read_csv('sjp2.csv')
# sjp44 = pd.read_csv('sjp44.csv')

# mh2 = pd.read_csv('mh2.csv')
# mh4 = pd.read_csv('mh4.csv')


# df = sjp44
# m = df[df['number of customers'] == 100000]['mean waiting time'].mean()
# s = df[df['number of customers'] == 100000]['mean waiting time'].std()

# print(f'{m:.4f}')
# print(f'{s:.4f}')
# print(f'{(m - 0.95*s):.4f}')
# print(f'{(m + 0.95*s):.4f}')










