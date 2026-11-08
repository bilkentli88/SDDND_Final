#!/usr/bin/env python3
"""Plot E4/E5 paired comparisons from the retained dataset-level records."""
from pathlib import Path
import argparse,csv,json,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'generated/reproduction/figures')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'studies/E5/results/dataset_pairs.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    conditions=sorted({(int(r['study']),float(r['noise_std']),float(r['observation_step']),float(r['rollout_step'])) for r in rows})
    fig,ax=plt.subplots(figsize=(8,4.4))
    for j,condition in enumerate(conditions):
        for control,shift,color in [('T1',-.13,'#2878b5'),('T2',.13,'#e1812c')]:
            selected=[r for r in rows if (int(r['study']),float(r['noise_std']),float(r['observation_step']),float(r['rollout_step']))==condition and r['metric']=='forecast_mse' and r['numerator']=='P3' and r['denominator']==control]
            ratios=[float(r['numerator_value'])/float(r['denominator_value']) for r in selected]
            assert len(ratios)==5
            ax.scatter(ratios,[j+shift]*5,color=color,s=18,alpha=.65,label='P3/'+control if j==0 else None)
            ax.scatter([statistics.median(ratios)],[j+shift],color=color,marker='D',s=50)
    labels=[f'E{n}, SD {s:g}'+(f', obs. {o:g}, h {h:g}' if n==3 else '') for n,s,o,h in conditions]
    ax.set_yticks(range(len(conditions)),labels);ax.invert_yaxis();ax.set_xscale('log');ax.axvline(1,color='.4',ls='--',lw=1)
    ax.set_xlabel('Paired forecast MSE ratio (P3 / two-stage control)');ax.legend(loc='lower left');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(a.output/f'two_stage_paired.{ext}',dpi=200)
    plt.close(fig)
    summary=json.loads((ROOT/'studies/E4/results/summary.json').read_text())
    blocks=summary['dataset_medians'];fig,axes=plt.subplots(1,2,figsize=(9,4))
    conditions=[(m,s) for m in ('External-C','Structured-C') for s in (.0005,.0015)]
    for ax,metric,title in zip(axes,['absolute_delay_error','h0.00625_test_clean_mse'],['Absolute delay error','Clean forecast MSE']):
        for j,(method,sigma) in enumerate(conditions):
            lookup={(b['method'],b['data_seed']):b[metric] for b in blocks if b['noise_std']==sigma}
            seeds=sorted({b['data_seed'] for b in blocks if b['noise_std']==sigma})
            values=[lookup[(method+'+anchors',seed)]/lookup[(method,seed)] for seed in seeds]
            assert len(values)==5
            ax.scatter(np.arange(5)*.045+j-.09,values,s=24)
            ax.plot([j-.18,j+.18],[statistics.median(values)]*2,color='black',lw=2)
        ax.set_yscale('log');ax.axhline(1,color='.4',ls='--',lw=1);ax.set_title(title)
        ax.set_xticks(range(4),[f'{m.replace("-C", "")}\nSD {s:g}' for m,s in conditions],fontsize=8)
        ax.set_ylabel('Anchored / unanchored ratio')
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(a.output/f'constant_delay_paired.{ext}',dpi=200)
    print('Generated E4/E5 paired plots in',a.output)

if __name__=='__main__':main()
