import os,sys
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from mebocost_test2 import mebocost
import importlib
importlib.reload(mebocost)

## pathway models
from mebocost_test2 import cscore
importlib.reload(cscore)

from mebocost_test2 import pyscenic_aucell as aucell
importlib.reload(aucell)

from ctxcore.genesig import GeneSignature

from scipy.sparse import csr_matrix, issparse

import collections

from scipy.stats import spearmanr, pearsonr

## try visionpy
from mebocost_test2 import visionpy
importlib.reload(visionpy)


def aucell_caculator(adata, 
                     gmt_path=[],
                     commu_res = pd.DataFrame(),
                     AUC_threshold = 0.05,
                     num_workers = 4,
                     cg_col = 'cell_group'):
    # gmt_path = ['./c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt']
    try:
        adata = adata.raw.to_adata()
    except:
        pass
    gmt_df = visionpy.signatures_from_gmt(gmt_files=gmt_path, adata=adata, use_raw = False)
    gmt_dict = {x: gmt_df[x][gmt_df[x] == 1].index.tolist() for x in gmt_df.columns.tolist()}

    ## covert gmt to GeneSignature format
    test_gmt=[]
    for i in gmt_dict.keys():
        test_gmt.append(GeneSignature(name=i,gene2weight=dict(zip(gmt_dict[i],[1 for i in gmt_dict[i]]))))
    ## prepare exp df
    try:
        matrix = adata.raw.to_adata().to_df()
    except:
        matrix = adata.to_df()
    
    ## run AUCell
    percentiles = aucell.derive_auc_threshold(matrix)
    auc_threshold = percentiles[AUC_threshold]
    
    aucs_mtx = aucell.aucell(matrix, signatures=test_gmt, auc_threshold=auc_threshold, num_workers=num_workers)
    
    ## save to adata to save RAM
    aucs_adata = sc.AnnData(X = aucs_mtx, obs = adata.obs)
    
    ## avg pathway scores in cell groups
    # avg_aucs=aucs_mtx.groupby(adata.obs[cg_col]).mean()
    del aucs_mtx
    del matrix
    
    ## sensor harboring pathway activity
    sensor_cont_pathway = {}
    if commu_res.shape[0] > 0:
        for s in commu_res['Sensor'].unique().tolist():
            tpath = [x for x in gmt_dict if s in gmt_dict[x]]
            if len(tpath) > 0:
                sensor_cont_pathway[s] = tpath

    ## compute correlation between sensor harboring pathway with other pathway scores
    sensor_cont_corr_pathway = {}
    if commu_res.shape[0] > 0:
        for s, r in commu_res[['Sensor', 'Receiver']].drop_duplicates().values.tolist():
            # print(s, r)
            if s not in sensor_cont_pathway:
                continue
            if s in sensor_cont_corr_pathway:
                sensor_cont_corr_pathway[s][r] = collections.defaultdict()
            else:
                sensor_cont_corr_pathway[s] = collections.defaultdict()
                sensor_cont_corr_pathway[s][r] = collections.defaultdict()
            dat = aucs_adata[aucs_adata.obs[cg_col] == r].X
            if issparse(dat):
                dat = dat.toarray()     
            corr_df = np.corrcoef(dat.T)
            cpath = sensor_cont_pathway[s]
            for p in cpath:
                tmp = corr_df[np.where(aucs_adata.var_names == p)[0][0],:] ## pathway vs. all other corr
                tmp[np.where(aucs_adata.var_names == p)[0][0]] = np.nan ## force itself to nan, because itself will be 1.
                sensor_cont_corr_pathway[s][r][p] = tmp
            ## return data structure:
            ## dict: {sensor: {receiver: {pathway: corrcoef}}}
    return(aucs_adata, sensor_cont_pathway, sensor_cont_corr_pathway)

def vision_caculator(adata, 
                     gmt_path=[],
                     commu_res = pd.DataFrame(),
                     cg_col = 'cell_group'):
    # gmt_path = ['./c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt']
    try:
        adata = adata.raw.to_adata()
    except:
        pass
    gmt_df = visionpy.signatures_from_gmt(gmt_files=gmt_path, adata=adata, use_raw = False)
    gmt_dict = {x: gmt_df[x][gmt_df[x] == 1].index.tolist() for x in gmt_df.columns.tolist()}

    adata.varm["signatures"] = gmt_df

    vis_df = visionpy.compute_signatures_anndata(adata = adata, 
                                        signature_varm_key = 'signatures',
                                        signature_names_uns_key = None,
                                        norm_data_key = None)
    ## save to adata to save RAM
    vis_adata = sc.AnnData(X = vis_df, obs = adata.obs)
    
    del vis_df
    del gmt_df

    ## sensor harboring pathway activity
    sensor_cont_pathway = {}
    if commu_res.shape[0] > 0:
        for s in commu_res['Sensor'].unique().tolist():
            tpath = [x for x in gmt_dict if s in gmt_dict[x]]
            if len(tpath) > 0:
                sensor_cont_pathway[s] = tpath

    ## compute correlation between sensor harboring pathway with other pathway scores
    sensor_cont_corr_pathway = {}
    if commu_res.shape[0] > 0:
        for s, r in commu_res[['Sensor', 'Receiver']].drop_duplicates().values.tolist():
            # print(s, r)
            if s not in sensor_cont_pathway:
                continue
            if s in sensor_cont_corr_pathway:
                sensor_cont_corr_pathway[s][r] = collections.defaultdict()
            else:
                sensor_cont_corr_pathway[s] = collections.defaultdict()
                sensor_cont_corr_pathway[s][r] = collections.defaultdict()
            dat = vis_adata[vis_adata.obs[cg_col] == r].X
            if issparse(dat):
                dat = dat.toarray()     
            corr_df = np.corrcoef(dat.T)
            cpath = sensor_cont_pathway[s]
            for p in cpath:
                tmp = corr_df[np.where(vis_adata.var_names == p)[0][0],:] ## pathway vs. all other corr
                tmp[np.where(vis_adata.var_names == p)[0][0]] = np.nan ## force itself to nan, because itself will be 1.
                sensor_cont_corr_pathway[s][r][p] = tmp
            sensor_cont_corr_pathway[s][r]['average_pathway_activity'] = dat.mean(axis = 0) 

            ## return data structure:
            ## dict: {sensor: {receiver: {pathway: corrcoef}}}
    return(vis_adata, sensor_cont_pathway, sensor_cont_corr_pathway)

def SensorPathwayPlot(pathy_adata,
                     sensor_cont_pathway,
                     sensor,
                     cg_col = 'cell_group',
                     receiver = [],
                     cg_focus = [],
                     pdf = None,
                      show_plot = False,
                      return_fig = False,
                      figsize = None):
    
    if sensor in sensor_cont_pathway:
        if cg_focus:
            pathy_adata = pathy_adata[pathy_adata.obs[cg_col].isin(cg_focus)]
            
        if receiver:
            tmp_adata = pathy_adata.copy()
            tmp_adata.obs['new_label'] = list(map(lambda x: x if x in receiver else 'Non-Receiver', pathy_adata.obs[cg_col]))
            fig = sc.pl.stacked_violin(adata = tmp_adata, var_names=sensor_cont_pathway[sensor],
                                   groupby='new_label', swap_axes=True, return_fig=True, figsize = figsize)
            fig.color_legend_title = 'Median Score\n in group'

        else:
            fig = sc.pl.stacked_violin(adata = pathy_adata, var_names=sensor_cont_pathway[sensor],
                                   groupby=cg_col, swap_axes=True, return_fig=True, figsize = figsize)
            fig.color_legend_title = 'Median Score\n in group'

        fig.show() if show_plot else None
        if pdf:
            pdf.savefig(fig)
        if return_fig:
            return(fig)
        return
    return
    


def PathwayScatterPlot(pathy_adata,
                      pathw_1,
                      pathw_2,
                      receiver,
                      cg_col = 'cell_group',
                      pdf = None,
                      show_plot = False,
                      return_fig = False,
                      figsize = (4.5, 4)
                      ):
    ## plot corr scatter
    # pathw_1 = sensor_cont_pathway[sensor][0]
    # pathw_2 = 'kegg_vascular_smooth_muscle_contraction'
    # receiver = 'VSM'
    # cg_col = 'cell_type'
    plot_df = sc.get.obs_df(pathy_adata[pathy_adata.obs[cg_col] == receiver], keys = [pathw_1, pathw_2])
    
    fig, ax = plt.subplots(figsize = figsize)
    sns.regplot(data = plot_df, x = pathw_1, y = pathw_2,
                line_kws={'color':'grey'}, scatter_kws={'s': 3})
    r, p = pearsonr(plot_df[pathw_1], plot_df[pathw_2])
    ax.set(xlabel='Pathway_1', ylabel = 'Pathway_2')
    ax.set_title('Corr Coeff=%.2f, p=%.2e'%(r, p))
    plt.tight_layout()
    plt.show() if show_plot else None
    plt.close()
    if return_fig:
        return(fig)
        
    


    
