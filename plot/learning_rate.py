import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
import random
from matplotlib.font_manager import FontProperties

plt.rcParams['figure.figsize'] = (6.0, 6.0)
#对比两天内同一时刻温度的变化情况
font = {'family': 'Arial',
        'size': 24}
matplotlib.rcParams['mathtext.rm'] = 'arial'
matplotlib.rcParams['font.family'] = ['Arial']
matplotlib.rc('font', **font)

poison_rate=[0.01,0.02,0.03,0.04,0.05]
x = poison_rate
y1 = random.sample(range(10, 30), 5)
y2 = random.sample(range(10, 30), 5)
y3 = random.sample(range(10, 30), 5)

plt.figure(figsize=(6, 6))
plt.figure().set_size_inches(6,6)
plt.xlabel('Poison Rate',fontdict=font)  # x轴标题
plt.ylabel('ASR',fontdict=font)  # y轴标题
plt.plot(x, y1, 
        #  color = 'darkblue',
         linestyle = '--',
         linewidth = 4,
         marker = 's',
         markersize = 15,
         #markeredgecolor = 'b',
         #markerfacecolor = 'r')   
) 
plt.plot(x, y2, 
        #  color = 'darkorange',
         linestyle = '--',
         linewidth = 4,
         marker = 's',
         markersize = 15,
         #markeredgecolor = 'b',
         #markerfacecolor = 'r')
)
plt.plot(x, y3, 
        #  color = 'darkorange',
         linestyle = '--',
         linewidth = 4,
         marker = 's',
         markersize = 15,
         #markeredgecolor = 'b',
         #markerfacecolor = 'r')
)
plt.xticks(poison_rate,size = 24)
plt.yticks(size = 24)
plt.grid(linestyle="-")
     

#绘制图例
plt.legend(['Dynamic','BadNL','Back_NLG'],fontsize=30,prop={'size':24},loc=4)
plt.hlines(4.05, 0.01, 0.05, linewidth = 3,color = 'grey',linestyles ='--')
     
# plt.ylim(3.8,4.2)

plt.savefig('./plot/learning_rate.pdf', dpi=300,bbox_inches='tight',  pad_inches = 0)
plt.show()
# plt.figure(figsize=(6, 6))#figsize=(6, 6)
# plt.figure().set_size_inches(6,6)
# # plt.subplot(1, 2, 2)

