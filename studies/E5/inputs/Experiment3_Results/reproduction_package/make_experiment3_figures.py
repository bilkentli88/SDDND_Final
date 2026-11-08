"""Static scientific figures: all five paired datasets, without significance labels."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,FixedLocator,NullLocator
from matplotlib.lines import Line2D


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    pairs=pd.read_csv(a.results/'dataset_pairs.csv');a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(2,2,figsize=(10,7.5),sharex='col')
    fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.94,hspace=.10,wspace=.28)
    for col,kind in enumerate(['solver_refinement','observation_refinement']):
        for row,metric in enumerate(['delay_rmse','refined_clean_test_mse']):
            ax=axes[row,col]
            subset=pairs[pairs.contrast.eq(kind)&pairs.metric.eq(metric)&pairs.method.isin(['P3','P4'])]
            labels=list(dict.fromkeys(subset.label))
            for j,label in enumerate(labels):
                g=subset[subset.label.eq(label)].sort_values('data_seed')
                assert len(g)==5 and g.complete.all()
                jitter=np.linspace(-.12,.12,5)
                for k,(_,r) in enumerate(g.iterrows()):
                    ax.scatter(j+jitter[k],r.ratio,s=40,color=colors[k],alpha=.9,
                               label=f'Dataset {k+1}' if row==col==0 and j==0 else None,zorder=3)
                median=float(g.ratio.median())
                ax.plot([j+.18,j+.32],[median,median],color='black',lw=2.3,zorder=4)
            ax.axhline(1,color='#666666',ls='--',lw=1,zorder=1)
            ax.set_yscale('log');ax.grid(axis='y',which='major',alpha=.18)
            ticks={(0,0):[.98,1,1.02,1.04,1.06,1.08,1.10],
                   (0,1):[.6,.7,.8,.9,1,1.2],
                   (1,0):[.95,1,1.05,1.10,1.15,1.20],
                   (1,1):[.9,1,1.1,1.2]}[(row,col)]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'))
            ax.set_xticks(range(len(labels)),[x.replace(', ','\n') for x in labels])
            ax.set_xlim(-.4,len(labels)-.5+.4)
            ax.set_ylabel('Fine / coarse delay RMSE' if row==0 else 'Fine / coarse held-out MSE')
            if row==0:ax.set_title('Training-solver refinement' if col==0 else 'Observation refinement')
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='black',lw=2.3));labels.append('Median')
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.015),ncol=6,frameon=False,fontsize=9,columnspacing=1)
    fig.savefig(a.output/'ASOC_Experiment3_Numerical_Sensitivity.png',dpi=200)
    fig.savefig(a.output/'ASOC_Experiment3_Numerical_Sensitivity.svg')
    plt.close(fig)


if __name__=='__main__':main()
