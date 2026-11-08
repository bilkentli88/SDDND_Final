"""Plot the prespecified paired estimand; no intervals or significance labels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
frame=pd.read_csv(args.results/'dataset_pairs.csv')
frame=frame[frame.complete & frame.ratio.notna() & frame.ratio.gt(0)]
comparators=['A1','A2','P4','Linear','PCHIP']
levels=sorted(frame.noise_std.unique())
seeds=sorted(frame.data_seed.unique())
palette=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
fig,axes=plt.subplots(1,len(levels),figsize=(10.2,4.5),sharey=True,squeeze=False)
minimum=min(.8,float(frame.ratio.min()))*.75
maximum=max(1.,float(frame.ratio.max()))*1.3
for axis,sigma in zip(axes[0],levels):
    subset=frame[frame.noise_std.eq(sigma)]
    axis.axhline(1.,color='#525252',linewidth=1.2)
    axis.axhline(.8,color='#777777',linewidth=1.,linestyle='--')
    for x,method in enumerate(comparators):
        group=subset[subset.comparator.eq(method)]
        for j,seed in enumerate(seeds):
            value=group[group.data_seed.eq(seed)].ratio
            if len(value):
                axis.scatter(x+(j-2)*.065,float(value.iloc[0]),s=30,color=palette[j],edgecolors='white',linewidths=.45,zorder=3)
        if len(group):
            axis.scatter(x+.23,float(group.ratio.median()),marker='D',s=32,color='#111111',edgecolors='white',linewidths=.4,zorder=4)
    axis.set_xticks(range(len(comparators)),comparators)
    axis.set_title(f'Observation noise SD = {sigma:g}',fontsize=11,pad=10)
    axis.set_yscale('log');axis.set_ylim(minimum,maximum)
    ticks=[value for value in (.1,.2,.5,.8,1.,2.,5.) if minimum<=value<=maximum]
    axis.set_yticks(ticks,[f'{value:g}' for value in ticks])
    axis.grid(axis='y',which='major',alpha=.16)
    axis.spines[['top','right']].set_visible(False)
    axis.set_xlabel('Comparator',labelpad=8)
axes[0,0].set_ylabel('Paired delay-RMSE ratio: P3 / comparator')
legend=[Line2D([],[],marker='o',color='none',markerfacecolor=palette[j],markeredgecolor='white',markersize=6,label=f'Dataset {j+1}') for j in range(len(seeds))]
legend.append(Line2D([],[],marker='D',color='none',markerfacecolor='#111111',markersize=5,label='Median of 5 ratios'))
fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.065),ncol=6,frameon=False,fontsize=8.3)
fig.text(.5,.025,'Below 1 favors P3. Dashed line: fixed 0.80 practical threshold. Five paired base-noise replications per condition.',ha='center',fontsize=8.5,color='#444444')
fig.subplots_adjust(left=.08,right=.985,top=.88,bottom=.29,wspace=.12)
args.output.mkdir(parents=True,exist_ok=True)
for suffix in ('png','svg'):
    path=args.output/f'ASOC_Experiment2_Paired_Ratios.{suffix}'
    fig.savefig(path,dpi=300,facecolor='white')
    print(path.name)
plt.close(fig)
