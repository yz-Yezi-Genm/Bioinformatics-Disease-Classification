"""Generate publication-style static figures from real selection/CV outputs."""
import argparse
import itertools
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

COLORS={'rf':'#245d9c','svm':'#c46b27'}

def save(fig,path):
    fig.savefig(path,dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',default='results/feature_selection')
    p.add_argument('--figures-dir',default='results/figures')
    args=p.parse_args()
    results,figures=Path(args.results_dir),Path(args.figures_dir)
    figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    summary=pd.read_csv(results/'selection_summary.csv')
    row=summary.iloc[0]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    bars=axes[0].bar(['Input','Nonconstant'],[row.input_features,row.after_variance],color=['#9ea9b5','#245d9c'],width=.6)
    axes[0].bar_label(bars,labels=[f'{int(row.input_features):,}',f'{int(row.after_variance):,}'],padding=5)
    axes[0].set_ylim(0,71000); axes[0].set_ylabel('Number of genes'); axes[0].set_title('1. Remove constant features')
    bars=axes[1].bar([f'k = {k}' for k in summary.k],summary.selected_features,color=['#245d9c','#468ab4','#78b3b5'],width=.6)
    axes[1].bar_label(bars,padding=5,fmt='%d'); axes[1].set_ylim(0,1200)
    axes[1].set_ylabel('Number of selected genes'); axes[1].set_title('2. Select top-scoring features')
    fig.suptitle('TCGA-BRCA | Feature selection on the full training set',fontsize=14)
    fig.supxlabel('976 samples, 877 patients | Descriptive fit; CV selectors are refitted within each training fold',fontsize=9)
    save(fig,figures/'feature_count_comparison.png')

    scores=pd.read_csv(results/'feature_scores.csv').query('passed_variance').nlargest(20,'score').sort_values('score')
    fig,ax=plt.subplots(figsize=(10,8),layout='constrained')
    ax.barh(scores.ensembl_id,scores.score,color='#245d9c',height=.7)
    ax.set_xlabel('ANOVA F score (higher means stronger univariate separation)')
    ax.set_title('Top 20 candidate gene features | Full training set',pad=15)
    ax.xaxis.grid(True,alpha=.18); ax.set_axisbelow(True)
    fig.supxlabel('Ranking evidence only; these genes are not validated diagnostic biomarkers',fontsize=10)
    save(fig,figures/'top_feature_scores.png')

    cv=pd.read_csv(results/'k_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,metric,title in zip(axes,['roc_auc','balanced_accuracy'],['ROC-AUC','Balanced accuracy']):
        for model,label in [('rf','Random Forest'),('svm','SVM (RBF)')]:
            sub=cv[cv.model==model].sort_values('k')
            x=np.arange(len(sub)) + (-.025 if model=='rf' else .025)
            ax.errorbar(x,sub[metric+'_mean'],yerr=sub[metric+'_std'],marker='o',capsize=4,
                        color=COLORS[model],label=label,linewidth=1.6)
        lower=float((cv[metric+'_mean']-cv[metric+'_std']).min())
        ax.set_ylim(max(0,lower-.02),1.015)
        ax.set_xticks(np.arange(len(summary)),[str(k) for k in sorted(summary.k)])
        ax.set_xlabel('Number of selected genes (k)'); ax.set_ylabel(title)
        ax.set_title(title); ax.grid(axis='y',alpha=.18); ax.legend(loc='lower right')
    fig.suptitle('Training-only reference comparison | 5 patient-grouped folds',fontsize=14)
    fig.supxlabel('Mean ± sample SD across folds; fixed model settings; these are not held-out test results',fontsize=9)
    save(fig,figures/'k_comparison_reference.png')

    selected=pd.read_csv(results/'cv_selected_features.csv')
    stability=[]
    for k,frame in selected.groupby('k'):
        sets=[set(g.feature_index) for _,g in frame.groupby('fold')]
        pairwise=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
        stability.append(dict(k=int(k),genes_selected_in_all_five_folds=len(set.intersection(*sets)),
                              unique_genes_across_folds=len(set.union(*sets)),mean_pairwise_jaccard=float(np.mean(pairwise))))
    pd.DataFrame(stability).to_csv(results/'selection_stability.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    k=int(summary.k.min())
    frequency=selected[selected.k==k].groupby('ensembl_id').fold.nunique().value_counts().reindex(range(1,6),fill_value=0)
    bars=ax.bar(frequency.index,frequency.values,color='#468ab4')
    ax.bar_label(bars,padding=4); ax.set_ylim(0,max(frequency.values)*1.2)
    ax.set_xticks(range(1,6)); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Number of training folds selecting the gene'); ax.set_ylabel('Number of distinct genes')
    ax.set_title(f'Selection stability across grouped CV | k = {k}')
    fig.supxlabel('Descriptive overlap; does not prove biological validity or independent replication',fontsize=9)
    save(fig,figures/'feature_selection_stability.png')
    print('Generated four PNG figures and selection_stability.csv')

if __name__=='__main__':
    main()
