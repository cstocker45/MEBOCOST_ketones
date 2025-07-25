#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: May 2022
# ================================

import os,sys
import time, re
import pickle as pk
import _pickle as cPickle
from datetime import datetime
import numpy as np
import pandas as pd
from operator import itemgetter
import scipy
from scipy import sparse
import scanpy as sc
import collections
import multiprocessing
import configparser
import tracemalloc
import warnings

from matplotlib.backends.backend_pdf import PdfPages

import importlib

import mebocost_test2.MetEstimator as ME
import mebocost_test2.crosstalk_calculator as CC
importlib.reload(CC)
import mebocost_test2.crosstalk_plots as CP
importlib.reload(CP)

# import mebocost_test2.pathway_enrichment as PE
# import mebocost_test2.pathway_plot as PP
import mebocost_test2.fba_handler as FBA
importlib.reload(FBA)

import mebocost_test2.crosstalk_diff as CD
importlib.reload(CD)

import mebocost_test2.pathway_associate as PA
importlib.reload(PA)

"""
linking input and out 
"""

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))


def _correct_colname_meta_(scRNA_meta, cellgroup_col=[]):
    """
    sometime the column names have different
    """
#     print(scRNA_meta)
    if scRNA_meta is None or scRNA_meta is pd.DataFrame:
        raise KeyError('Please provide cell_ann data frame!')
    
    if cellgroup_col:
        ## check columns names
        for x in cellgroup_col:
            if x not in scRNA_meta.columns.tolist():
                info('ERROR: given cell group identifier {} is not in meta table columns'.format(x))
                raise ValueError('given cell group identifier {} is not in meta table columns'.format(x))
        ## get cell group name
        scRNA_meta['cell_group'] = scRNA_meta[cellgroup_col].astype('str').apply(lambda row: '_'.join(row), axis = 1).tolist()
    else:
        info('no cell group given, try to search cluster and cell_type')
        col_names = scRNA_meta.columns.tolist()
        if 'cell_type' in col_names:
            pass
        elif 'cell_type' not in col_names and 'Cell_Type' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL_TYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'celltype' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELLTYPE' else x for x in col_names]
        elif 'cell_type' not in col_names and 'CellType' in col_names:
            scRNA_meta.columns = ['cell_type' if x.upper() == 'CELL TYPE' else x for x in col_names]
        else:
            info('ERROR: "cell_type" not in scRNA meta column names, will try cluster')
            if 'cluster' not in col_names and 'Cluster' in col_names:
                scRNA_meta.columns = ['cluster' if x.upper() == 'CLUSTER' else x for x in col_names]
            else:
                raise KeyError('cluster cannot find in the annotation, and cell_group does not specified')
            raise KeyError('cell_type cannot find in the annotation, and cell_group does not specified'.format(x))
        
        if 'cell_type' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cell_type'].tolist()
        elif 'cluster' in scRNA_meta.columns.tolist():
            scRNA_meta['cell_group'] = scRNA_meta['cluster'].tolist()
        else:
            raise KeyError('Please a group_col to group single cell')
    return(scRNA_meta)

def _read_config(conf_path):
    """
    read config file
    """
    #read config
    cf = configparser.ConfigParser()
    cf.read(conf_path)
    config = cf._sections
    # remove the annotation:
    for firstLevel in config.keys():
        for secondLevel in config[firstLevel]:
            if '#' in config[firstLevel][secondLevel]:
                config[firstLevel][secondLevel] = config[firstLevel][secondLevel][:config[firstLevel][secondLevel].index('#')-1].rstrip()
    return(config)


def load_obj(path):
    """
    read mebocost object
    """
    try:
        file = open(path,'rb')
        dataPickle = file.read()
        file.close()
        obj_vars = cPickle.loads(dataPickle)
    except:
        obj_vars = pd.read_pickle(path)
    ## check for group_col: for v1 it accepts list, for v2, only string
    ## so need to check
    if isinstance(obj_vars['group_col'], list):
        if 'cell_group' in obj_vars:
            obj_vars['group_col'] = "cell_group"
        else:
            obj_vars['cell_ann']['cell_group'] = obj_vars['cell_ann'][obj_vars['group_col'][0]].astype(str)+' ~ '+obj_vars['cell_ann'][obj_vars['group_col'][1]].astype(str)
            obj_vars['group_col'] = "cell_group"
            
    mebocost_obj = create_obj(exp_mat = obj_vars['exp_mat'] if 'exp_mat' in obj_vars else None,
                        adata = obj_vars['adata'] if 'adata' in obj_vars else None,
                        cell_ann = obj_vars['cell_ann'] if 'cell_ann' in obj_vars else None,
                        group_col = obj_vars['group_col'] if 'group_col' in obj_vars else None,
                        condition_col = obj_vars['condition_col'] if 'condition_col' in obj_vars else None,
                        config_path = obj_vars['config_path'] if 'config_path' in obj_vars else None,
                       )
    mebocost_obj.__dict__ = obj_vars

    return mebocost_obj


def save_obj(obj, path = 'mebocost_result.pk', filetype = 'pickle'):
    """
    save object to pickle
    """
    file = open(path, 'wb')
    file.write(cPickle.dumps(obj.__dict__))
    file.close()

def _check_exp_mat_(exp_mat):
    """
    check if the expression matrix are all numerical
    """
    str_cols = exp_mat.apply(lambda col: np.array_equal(col, col.astype(str)))
    str_rows = exp_mat.apply(lambda row: np.array_equal(row, row.astype(str)), axis = 1)
    if np.any(str_cols == True):
        warnings.warn("%s column is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    if np.any(str_rows == True):
        warnings.warn("%s row is a str, will be removed as only int or float accepted in expression matrix"%(exp_mat.columns[str_cols]))
    exp_mat = exp_mat.loc[~str_rows, ~str_cols]
    return(exp_mat)


class create_obj:
    """
    MEBOCOST for predicting metabolite-based cell-cell communication (mCCC). The modules of the package include communication inference, communication visualization, pathway inference, pathway visualization.

    Params
    -------
    exp_mat
        python pandas data frame, single cell expression matrix, rows are genes, columns are cells
        'exp_mat' is a exclusive parameter to 'adata'
    adata
        scanpy adata object, the expression will be extracted, 'adata' is an exclusive parameter to 'exp_mat'
    cell_ann
        data frame, cell annotation information, cells in row names
    group_col
        a list, specify the column names in 'cell_ann' for grouping cells, by default 'cell_type' or 'cluster' will be detected and used
    condition_col
        a list, specify the column names in 'cell_ann' for running mCCC in different samples/conditions, e.g., control, treatment
    species
        human or mouse, this determines which database will be used in our collection

    met_est
        the method for estimating metabolite enzyme expression in cell, should be one of:
        mebocost: estimated by the enzyme network related to the metabolite
        scFEA-flux: flux result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        scFEA-balance: balance result of published software scFEA (https://pubmed.ncbi.nlm.nih.gov/34301623/)
        compass-reaction: reaction result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-uptake: uptake result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
        compass-secretion: secretion result of published software Compass (https://pubmed.ncbi.nlm.nih.gov/34216539/)
    met_pred
        data frame, if scFEA or Compass is used to impute the metabolite enzyme expression in cells, please provide the original result from scFEA or Compass, cells in row names, metabolite/reaction/module in column names, 
        Noted that this parameter will be ignored if 'met_est' was set as mebocost.

    config_path
        str, the path for a config file containing the path of files for metabolite annotation, enzyme, sensor, scFEA annotation, compass annotation. These can also be specified separately by paramters as following:

        if config_path not given, please set:
    met_enzyme
        data frame, metabolite and gene (enzyme) relationships, required columns include HMDB_ID, gene, direction, for instance:
        
        HMDB_ID     gene                                                direction
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003375 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   substrate
        HMDB0003450 Cyp2c54[Unknown]; Cyp2c38[Unknown]; Cyp2c50[Un...   product
        HMDB0003948 Tuba8[Unknown]; Ehhadh[Unknown]; Echs1[Enzyme]...   product

    met_sensor
        data frame, metabolite sensor information, each row is a pair of metabolite and sensor, must include columns  HMDB_ID, Gene_name, Annotation, for instance:
        
        HMDB_ID Gene_name   Annotation
        HMDB0006247 Abca1   Transporter
        HMDB0000517 Slc7a1  Transporter
        HMDB0000030 Slc5a6  Transporter
        HMDB0000067 Cd36    Transporter
        
    met_ann:
        data frame, the annotation of metabolite collected from HMDB website, these are basic annotation info including HMDB_ID, Kegg_ID, metabolite, etc

    scFEA_ann
        data frame, module annotation of metabolite flux in scFEA, usually is the file at https://github.com/changwn/scFEA/blob/master/data/Human_M168_information.symbols.csv

    compass_met_ann
        data frame, the metabolite annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/met_md.csv

    compass_rxn_ann
        data frame, the reaction annotation used in Compass software, usually is the file at https://github.com/YosefLab/Compass/blob/master/compass/Resources/Recon2_export/rxn_md.csv

    gene_network
        data frame, gene by gene matrix, the value represent the association between two genes, will be used to evaluate downstream effect of the communication

    gmt_path
        a path, this parameter can be provided in config file and given by config_path. Only set this when you do not pass config_path parameter in. The gmt file contains pathway gene list, will be used in pathway inference module, the details of GMT format could be found at https://software.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#:~:text=The+GMT+file+format+is,genes+in+the+gene+set. 

    cutoff_exp
        auto or float, used to filter out cells which are lowly expressed for the given gene, by default is auto, meaning that automatically decide cutoffs for sensor expression to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_met
        auto or float, used to filter out cells which are lowly abundant of the given metabolite, by default is auto, meaning that automatically decide cutoffs for metabolite presence to exclude the lowly 25% non-zeros across all sensor or metabolites in all cells in addition to zeros 

    cutoff_prop
        float from 0 to 1, used to filter out metabolite or genes if the proportion of their abundant cells less than the cutoff

    sensor_type
        a list, provide a list of sensor type that will be used in the communication modeling, must be one or more from ['Receptor', 'Transporter', 'Nuclear Receptor'], default is all the three

    thread
        int, number of cores used for running job, default 1
        
    """
    def __init__(self,  
                exp_mat=None, 
                adata=None, 
                cell_ann=None,
                group_col='celltype',
                condition_col=None,
                species = 'human',

                met_est=None,
                met_pred=pd.DataFrame(), 

                config_path=None,
                met_enzyme=pd.DataFrame(),
                met_sensor=pd.DataFrame(),
                met_ann=pd.DataFrame(), 
                scFEA_ann=pd.DataFrame(),
                compass_met_ann=pd.DataFrame(),
                compass_rxn_ann=pd.DataFrame(),
                gene_network=pd.DataFrame(),
                gmt_path = None,

                cutoff_exp='auto',
                cutoff_met='auto',
                cutoff_prop=0.15,

                sensor_type=['Receptor', 'Transporter', 'Nuclear Receptor'],
                thread = 1
                ):
        tic = time.time()

        self.exp_mat = exp_mat
        self.adata = adata
        ## check cell group information
        ## add a column "cell_group" if successfull
        if (self.exp_mat is None and cell_ann is None) and (self.adata is not None):
            cell_ann = adata.obs.copy()
        if group_col not in cell_ann.columns.tolist():
            raise KeyError('group_col: %s is not in cell_ann columns, it should be one of %s'%(group_col, cell_ann.columns.tolist()))
        else:
            self.group_col = group_col
            # cell_ann['cell_group'] = cell_ann[group_col].tolist()

        if not condition_col or condition_col in cell_ann.columns.tolist():
            self.condition_col = condition_col
        else:
            raise KeyError('condition_col: %s is not in cell_ann columns, it should be one of %s'%(condition_col, cell_ann.columns.tolist()))
        
        # self.cell_ann = _correct_colname_meta_(cell_ann, cellgroup_col = self.group_col)
        self.cell_ann = cell_ann
        self.species = species

        self.met_est = 'mebocost' if not met_est else met_est # one of [scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]
        self.met_pred = met_pred

        ## the path of config file
        self.config_path = config_path
        ## genes (enzyme) related to met
        self.met_enzyme = met_enzyme
        ## gene name in metaboltie sensor
        self.met_sensor = met_sensor
        ## met basic ann
        self.met_ann = met_ann
        ## software ann
        self.scFEA_ann = scFEA_ann
        self.compass_met_ann = compass_met_ann
        self.compass_rxn_ann = compass_rxn_ann
        ## gene network
        self.gene_network = gene_network
        if not self.config_path and (self.met_sensor is None or self.met_sensor.shape[0] == 0):
            raise KeyError('Please either provide config_path or a data frame of met_enzyme, met_sensor, met_ann, gene_network, etc')

        ## cutoff for expression, metabolite, and proportion of cells
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.cutoff_prop = cutoff_prop
        self.sensor_type = sensor_type
        self.gmt_path = gmt_path
        self.thread = thread
        self.mode = 'scrna'
        
        ## ============== initial ===========

        if self.exp_mat is None and self.adata is None:
            raise ValueError('ERROR: please provide expression matrix either from exp_mat or adata (scanpy object)')  
        elif self.exp_mat is None and self.adata is not None:
            ## check the adata object
            ngene = len(self.adata.var_names)
            ncell = len(self.adata.obs_names)
            info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            if ngene < 5000:
                info('scanpy object contains less than 5000 genes, please make sure you are using raw.to_adata()')
            self.exp_mat = sparse.csc_matrix(self.adata.X.T)
            self.exp_mat_indexer = self.adata.var_names
            self.exp_mat_columns = self.adata.obs_names
            self.adata = None
        else:
            if 'scipy.sparse' in str(type(self.exp_mat)):
                ## since the scipy version problem leads to the failure of using sparse.issparse
                ## use a simple way to check!!!
                #sparse.issparse(self.exp_mat):
                pass 
            elif type(self.exp_mat) is type(pd.DataFrame()):
                ## check if the exp_mat values are all int or float
                self.exp_mat = _check_exp_mat_(self.exp_mat)
                self.exp_mat_indexer = self.exp_mat.index ## genes
                self.exp_mat_columns = self.exp_mat.columns ## columns
                self.exp_mat = sparse.csc_matrix(self.exp_mat)
                ngene, ncell = self.exp_mat.shape
                info('We get expression data with {n1} genes and {n2} cells.'.format(n1 = ngene, n2 = ncell))
            else:
                info('ERROR: cannot read the expression matrix, please provide pandas dataframe or scanpy adata')
        if self.condition_col not in ['', False, 'NA', None, 'None']:
            group_names = self.cell_ann[self.condition_col].astype('str').str.replace('~', '_')+' ~ '+self.cell_ann[self.group_col].astype('str').str.replace('~', '_')
        else:
            group_names = self.cell_ann[self.group_col]
        self.cell_ann['cell_group'] = group_names.copy()
        self.group_names = group_names.unique().tolist()
        
        ## end preparation
        toc = time.time()
        info('Data Preparation Done in {:.4f} seconds'.format(toc-tic))


    def _load_config_(self):
        """
        load config and read data from the given path based on given species
        """
        ## the path of config file
        info('Load config and read data based on given species [%s].'%(self.species))
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise KeyError('ERROR: the config path is not exist!')
            config = _read_config(conf_path = self.config_path)
            ## common
            self.met_ann = pd.read_csv(config['common']['hmdb_info_path'], sep = '\t')
            if self.met_est.startswith('scFEA'):
                    self.scFEA_ann = pd.read_csv(config['common']['scfea_info_path'], index_col = 0)
            if self.met_est.startswith('compass'):
                self.compass_met_ann = pd.read_csv(config['common']['compass_met_ann_path'])
                self.compass_rxn_ann = pd.read_csv(config['common']['compass_rxt_ann_path'])
            ## depends on species
            if self.species == 'human':
                self.met_enzyme = pd.read_csv(config['human']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['human']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
                self.gmt_path = {'KEGG': config['human']['kegg_gmt_path'], 
                                 'Wikipathway': config['human']['wikipathway_gmt_path'], 
                                 'Reactome': config['human']['reactome_gmt_path']}
                self.gem_path = config['human']['gem_path']
            elif self.species == 'mouse':
                self.met_enzyme = pd.read_csv(config['mouse']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['mouse']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
                self.gmt_path = {'KEGG':config['mouse']['kegg_gmt_path'], 
                                 'Wikipathway': config['mouse']['wikipathway_gmt_path'], 
                                 'Reactome': config['mouse']['reactome_gmt_path']}
                self.gem_path = config['mouse']['gem_path']
            else:
                raise KeyError('Species should be either human or mouse!')
            ## check row and columns, we expect rows are genes, columns are cells
            if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) < 10:
                raise KeyError('it looks like that both the row and columns are not matching to gene name very well, please check the provided matrix or species!')
            if len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_indexer.tolist())) < 10 and len(set(self.met_sensor['Gene_name'].tolist()) & set(self.exp_mat_columns.tolist())) > 10:
                info('it is likely the columns of the exp_mat are genes, will transpose the matrix')
                self.exp_mat = self.exp_mat.T
                columns = self.exp_mat_indexer.copy()
                index = self.exp_mat_columns.copy()
                self.exp_mat_indexer = index
                self.exp_mat_columns = columns
        else:
            info('please provide config path')

    def estimator(self):
        """
        estimate of metabolite enzyme expression in cells using the expression of related enzymes
        """
        info('Estimtate metabolite enzyme expression using %s'%self.met_est)
        mtd = self.met_est

        if mtd == 'mebocost':
            met_mat, met_indexer, met_columns = ME._met_from_enzyme_est_(exp_mat=self.exp_mat, 
                                                   indexer = self.exp_mat_indexer,
                                                   columns = self.exp_mat_columns,
                                                    met_gene=self.met_enzyme, 
                                                    method = 'mean')
        elif mtd == 'scFEA-flux':
            met_mat = ME._scFEA_flux_est_(scFEA_pred = self.met_pred, 
                                            scFEA_info=self.scFEA_ann, 
                                            hmdb_info=self.met_ann)
        elif mtd == 'scFEA-balance':
            met_mat = ME._scFEA_balance_est_(scFEA_pred = self.met_pred, 
                                                scFEA_info=self.scFEA_ann, 
                                                hmdb_info=self.met_ann)
        elif mtd == 'compass-reaction':
            met_mat = ME._compass_react_est_(compass_pred=self.met_pred, 
                                                compass_react_ann=self.compass_rxn_ann, 
                                                compass_met_ann=self.compass_met_ann, 
                                                hmdb_info=self.met_ann)
        else:
            raise KeyError('Please specify "met_est" to be one of [mebocost, scFEA-flux, scFEA-balance, compass-reaction, compass-uptake, compass-secretion]')
        
        self.met_mat = sparse.csc_matrix(met_mat)
        self.met_mat_indexer = np.array(met_indexer)
        self.met_mat_columns = np.array(met_columns)
#         return met_mat


    def infer(self, met_mat=pd.DataFrame(), n_shuffle = 1000, seed = 12345, thread = None):
        """
        excute communication prediction
        met_mat
            data frame, columns are cells and rows are metabolites
        """
        info('Infer communications')
        if met_mat.shape[0] != 0: ## if given met_mat in addition
            self.met_mat_indexer = np.array(met_mat.index)
            self.met_mat_columns = np.array(met_mat.columns)
            self.met_mat = sparse.csc_matrix(met_mat)
        ## focus on met and gene of those are in the data matrix
        met_sensor = self.met_sensor[self.met_sensor['Gene_name'].isin(self.exp_mat_indexer) & 
                                     self.met_sensor['HMDB_ID'].isin(self.met_mat_indexer)]
        self.met_sensor = met_sensor

        ## init
        cobj = CC.InferComm(exp_mat = self.exp_mat,
                            exp_mat_indexer = self.exp_mat_indexer, 
                            exp_mat_columns = self.exp_mat_columns,
                            avg_exp = self.avg_exp,
                            avg_exp_indexer = self.avg_exp_indexer,
                            avg_exp_columns = self.avg_exp_columns,
                            met_mat = self.met_mat,
                            met_mat_indexer = self.met_mat_indexer,
                            met_mat_columns = self.met_mat_columns,
                            avg_met = self.avg_met,
                            avg_met_indexer = self.avg_met_indexer,
                            avg_met_columns = self.avg_met_columns,
                            cell_ann = self.cell_ann,
                            group_col = self.group_col,
                            condition_col = self.condition_col,
                            met_sensor = self.met_sensor,
                            sensor_type = self.sensor_type,
                            thread = thread
                           )

        commu_res_df, commu_res_bg = cobj.pred(n_shuffle = n_shuffle, seed = seed)
    
        ## add metabolite name
        hmdbid_to_met = {}
        for Id, met in self.met_ann[['HMDB_ID', 'metabolite']].values.tolist():
            hmdbid_to_met[Id] = met
        ## add name
        commu_res_df['Metabolite_Name'] = list(map(lambda x: hmdbid_to_met.get(x) if x in hmdbid_to_met else None,
                                                   commu_res_df['Metabolite']))

        ## add annotation
        sensor_to_ann = {}
        for s, a in self.met_sensor[['Gene_name', 'Annotation']].values.tolist():
            sensor_to_ann[s] = a
        commu_res_df['Annotation'] = list(map(lambda x: sensor_to_ann.get(x) if x in sensor_to_ann else None,
                                              commu_res_df['Sensor']))
        
        return commu_res_df, commu_res_bg


    def _filter_lowly_aboundant_(self, 
                                 pvalue_res,
                                 cutoff_prop,
                                 met_prop=None,
                                 exp_prop=None,
                                 pval_method='permutation_test_fdr',
                                 pval_cutoff=0.05,
                                 min_cell_number=50,
                                 return_signi_only = False
                                ):
        """
        change p value to 1 if either metabolite_prop or transporter_prop less than the cutoff 
        (meaning that no enough metabolite or sensor present in the cell group)
        -------
         pvalue_res,
         cutoff_prop,
         met_prop=None,
         exp_prop=None,
         pval_method='permutation_test_fdr',
         pval_cutoff=0.05,
         min_cell_number=50,
         return_signi_only = False
        """
        res = pvalue_res.copy()
        ## add the metabolite abudance proportion
        if met_prop is not None:
            res['metabolite_prop_in_sender'] = [met_prop.loc[s, m] for s, m in res[['Sender', 'Metabolite']].values.tolist()]
        ## add the metabolite abudance proportion
        if exp_prop is not None:
            res['sensor_prop_in_receiver'] = [exp_prop.loc[r, s] for r, s in res[['Receiver', 'Sensor']].values.tolist()]
        
        if 'original_result' not in list(vars(self)):
            self.original_result = res.copy()
        ## minimum cell number
        cell_count = pd.Series(dict(collections.Counter(self.cell_ann['cell_group'].tolist())))
        bad_cellgroup = cell_count[cell_count<min_cell_number].index.tolist() 
        
        info('Set p value and fdr to 1 if sensor or metaboltie expressed cell proportion less than {}'.format(cutoff_prop))
        bad_index = np.where((res['metabolite_prop_in_sender'] <= cutoff_prop) |
                             (res['sensor_prop_in_receiver'] <= cutoff_prop) |
                             (res['Commu_Score'] < 0) |
                             (res['Sender'].isin(bad_cellgroup)) | 
                             (res['Receiver'].isin(bad_cellgroup))
                            )[0]
        if len(bad_index) > 0:
            pval_index = np.where(res.columns.str.endswith('_pval'))[0]
            res.iloc[bad_index, pval_index] = 1 # change to 1
            fdr_index = np.where(res.columns.str.endswith('_fdr'))[0]
            res.iloc[bad_index, fdr_index] = 1 # change to 1

        ## norm communication score
        res['Commu_Score'] = res['Commu_Score']/np.array(res['bg_mean']).clip(min = 0.05)
        
        if return_signi_only:
            ## filter out non-significant pairs
            if pval_method in res.columns.tolist():
                res = res[res[pval_method]<pval_cutoff]
            else:
                warnings.warn('%s is not in pvalue_res table columns, so will skip p value filter'%pval_method)

        if 'Condition' in res.columns.tolist():
            ## reorder columns
            columns = ['Sender', 'Receiver', 'Condition',
                        'Metabolite', 'Metabolite_Name', 'Sensor', 
                        'Annotation', 'Commu_Score', 
                       'metabolite_prop_in_sender',
                       'sensor_prop_in_receiver', 
                       'ttest_stat','ttest_pval', 'ranksum_test_stat', 'ranksum_test_pval',
                       'permutation_test_stat', 'permutation_test_pval',
                       'ttest_fdr', 'ranksum_test_fdr',
                       'permutation_test_fdr']
        else:
            columns = ['Sender', 'Receiver',
                        'Metabolite', 'Metabolite_Name', 'Sensor', 
                        'Annotation', 'Commu_Score', 
                       'metabolite_prop_in_sender',
                       'sensor_prop_in_receiver', 
                        'ttest_stat','ttest_pval', 'ranksum_test_stat', 'ranksum_test_pval',
                       'permutation_test_stat', 'permutation_test_pval',
                       'ttest_fdr', 'ranksum_test_fdr',
                       'permutation_test_fdr']
        get_columns = [x for x in columns if x in res.columns.tolist()]
        res = res.reindex(columns = get_columns).sort_values('permutation_test_fdr')
        ## record updated parameters
        self.cutoff_prop = cutoff_prop
        self.pval_method = pval_method
        self.pval_cutoff = pval_cutoff
        self.min_cell_number = min_cell_number
        return(res)

    def _auto_cutoff_(self, mat, q = 0.25):
        """
        given a matrix, such as gene-by-cell matrix,
        find 25% percentile value as a cutoff
        meaning that, for example, sensor in cell with lowest 25% expression will be discarded, by default.
        """
        v = []
        for x in mat:
            if np.all(x.toarray() <= 0):
                continue
            xx = x.toarray()
            xx = xx[xx>0]
            v.extend(xx.tolist())
        v = np.array(sorted(v))
        c = np.quantile(v, q)
        return(c)


    def _check_aboundance_(self, cutoff_exp=None, cutoff_met=None):
        """
        check the aboundance of metabolite or transporter expression in cell clusters,
        return the percentage of cells that meet the given cutoff
        by default, cutoff for metabolite aboundance is 0, expression of transporter is 0
        """
        info('Calculating metabolite presence and sensor expression in cell groups')
        ## this will re-write the begin values
        j1 = cutoff_exp is None or cutoff_exp is False
        j2 = self.cutoff_exp is None or self.cutoff_exp is False
        j3 = self.cutoff_exp == 'auto'
        j4 = isinstance(self.cutoff_exp, float) or isinstance(self.cutoff_exp, int)

        if cutoff_exp == 'auto':
            # decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            sensor_loc = np.where(self.exp_mat_indexer.isin(self.met_sensor['Gene_name']))[0]
            sensor_mat = self.exp_mat[sensor_loc,:]
            cutoff_exp = self._auto_cutoff_(mat = sensor_mat)
            self.cutoff_exp = cutoff_exp
            info('automated cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j4:
            cutoff_exp = self.cutoff_exp 
            info('provided cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        elif j1 and j2:
            cutoff_exp = 0
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        else:
            cutoff_exp = 0 if not cutoff_exp else cutoff_exp
            info('cutoff for sensor expression, cutoff=%s'%cutoff_exp)
        ## met 
        j1 = cutoff_met is None or cutoff_met is False
        j2 = self.cutoff_met is None or self.cutoff_met is False
        j3 = self.cutoff_met == 'auto'
        j4 = isinstance(self.cutoff_met, float) or isinstance(self.cutoff_met, int)

        if cutoff_met == 'auto':
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and (j2 or j3):
            ## decide cutoff by taking 75% percentile across all sensor in all cells
            cutoff_met = self._auto_cutoff_(mat = self.met_mat)
            self.cutoff_met = cutoff_met
            info('automated cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and j4:
            cutoff_met = self.cutoff_met 
            info('provided cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        elif j1 and j2:
            cutoff_met = 0
            info('cutoff for metabolite presence, cutoff=%s'%cutoff_met)
        else:
            cutoff_met = 0 if not cutoff_met else cutoff_met
            info('cutoff for metabolite presence, cutoff=%s'%cutoff_met)

        ## expression for all transporters
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        info('cutoff_exp: {}'.format(cutoff_exp))
        
        sensor_loc = {g:i for i,g in enumerate(self.exp_mat_indexer) if g in sensors}
        exp_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.exp_mat_columns) if c in cells]
            s = self.exp_mat[list(sensor_loc.values()),:][:,cell_loc]
            exp_prop[x] = pd.Series([v[v>cutoff_exp].shape[1] / v.shape[1] for v in s],
                                   index = list(sensor_loc.keys()))
        exp_prop = pd.DataFrame.from_dict(exp_prop, orient = 'index')
         
        # ====================== #
        info('cutoff_metabolite: {}'.format(cutoff_met))
        ## metabolite aboundance
        metabolites = self.met_sensor['HMDB_ID'].unique().tolist()
        met_prop = {}
        for x in self.cell_ann['cell_group'].unique().tolist():
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = [i for i, c in enumerate(self.met_mat_columns) if c in cells]
            m = self.met_mat[:,cell_loc]
            met_prop[x] = pd.Series([v[v>cutoff_met].shape[1] / v.shape[1] for v in m],
                                   index = self.met_mat_indexer.tolist())
        met_prop = pd.DataFrame.from_dict(met_prop, orient = 'index')
        
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.exp_prop = exp_prop
        self.met_prop = met_prop ## cell_group x sensor gene, cell_group x metabolite
    
    def _get_gene_exp_(self):
        """
        only sensor and enzyme gene expression are needed for each cells
        """
        sensors = self.met_sensor['Gene_name'].unique().tolist()
        enzymes = []
        for x in self.met_enzyme['gene'].tolist():
            enzymes.extend([i.split('[')[0] for i in x.split('; ')])
        genes = list(set(sensors+enzymes))
        ## gene loc
        gene_loc = np.where(pd.Series(self.exp_mat_indexer).isin(genes))[0]
        
        gene_dat = self.exp_mat[gene_loc].copy()
        ## update the exp_mat and indexer
        self.exp_mat = sparse.csr_matrix(gene_dat)
        self.exp_mat_indexer = self.exp_mat_indexer[gene_loc]
                                   
    def _avg_by_group_(self):
        ## avg exp by cell_group for met sensor
        group_names = self.cell_ann['cell_group'].unique().tolist()
        avg_exp = np.empty(shape = (self.exp_mat.shape[0],0)) ## save exp data

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.exp_mat_columns).isin(cells))[0]
            # arithmatic mean
            avg_exp = np.concatenate((avg_exp, self.exp_mat[:,cell_loc].mean(axis = 1)), axis = 1)
        
        self.avg_exp = sparse.csr_matrix(avg_exp)
        self.avg_exp_indexer = np.array(self.exp_mat_indexer)
        self.avg_exp_columns = np.array(group_names)
    
    
    def _avg_met_group_(self):
        """
        take average of sensor expression and metabolite by cell groups
        """
        ## avg met by cell_group for met
        avg_met = np.empty(shape = (self.met_mat.shape[0],0)) ## save exp data
        group_names = self.cell_ann['cell_group'].unique().tolist()

        for x in group_names:
            cells = self.cell_ann[self.cell_ann['cell_group'] == x].index.tolist()
            cell_loc = np.where(pd.Series(self.met_mat_columns).isin(cells))[0]
            ## mean
            avg_met = np.concatenate((avg_met, self.met_mat[:,cell_loc].mean(axis = 1)), axis = 1)

        self.avg_met = sparse.csr_matrix(avg_met)
        self.avg_met_indexer = np.array(self.met_mat_indexer)
        self.avg_met_columns = group_names
            
## ============================== constrain by flux ============================
    def _matchMetName_(self, met_name_list = []):
        """
        met_name is a list of metabolite names to match with HMDB standard names
        """
        met_ann = mebo_obj.met_ann[['metabolite', 'synonyms_name']]
        met_syn_dict = {}
        ## build a dict key is alias and value is standard name
        for mn, sn in met_ann.dropna().values.tolist():
            for ms in sn.split('; '):
                met_syn_dict[ms] = mn
        ## match names
        met_matched = {m:met_syn_dict.get(m) for m in met_name_list if m in met_syn_dict}
        return(met_matched)
            
    def _ConstrainFluxFromAnyTool_(self,
                            efflux_mat,
                            influx_mat,
                            efflux_cut = 'auto', 
                            influx_cut = 'auto',
                            norm=False,
                            inplace=True
                            ):
        """
        constraint efflux and influx for mCCC events based FBA results from any tools
        efflux_mat and influx_mat should be provided as data frames with rows for metabolite names, columns for cell group
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        cg_all = list(set(comm_res['Sender'].tolist()+comm_res['Receiver'].tolist()))
        info('Match cell groups')
        influx_cg = influx_mat.columns.tolist()
        ## check if all cell group in mCCC table in influx or efflux table
        influx_cg_check = np.all([x in influx_mat.columns.tolist() for x in cg_all])
        efflux_cg = efflux_mat.columns.tolist()
        efflux_cg_check = np.all([x in efflux_mat.columns.tolist() for x in cg_all])
        if not (influx_cg_check and efflux_cg_check):
            raise KeyError('Error: the cell group in commu_res does match in efflux or influx matrix')
    
        info('Match metabolites')
        influx_met = influx_mat.index.tolist()
        efflux_met = efflux_mat.index.tolist()
        influx_met_match = _matchMetName_(met_name_list = influx_met)
        efflux_met_match = _matchMetName_(met_name_list = efflux_met)
        if len(influx_met_match) == 0:
            raise KeyError('Error: No metabolite in influx matrix can match with commu_res')
        if len(efflux_met_match) == 0:
            raise KeyError('Error: No metabolite in efflux matrix can match with commu_res')
        ## rename efflux and influx matrix by standard name
        met_known = self.met_ann['metabolite'].unique().tolist()
        efflux_mat.index = [efflux_met_match.get(x, x) if x not in met_known else x for x in efflux_mat.index.tolist()]
        influx_mat.index = [influx_met_match.get(x, x) if x not in met_known else x for x in influx_mat.index.tolist()]
        ## concate to commu_res
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        if norm:
            flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
            comm_res[x1] = [flux_norm(efflux_mat.loc[m,c].max()) if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
            comm_res[x2] = [flux_norm(influx_mat.loc[m,c].max()) if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
            if efflux_cut == 'auto':
                all_efflux = [flux_norm(efflux_mat.loc[m,c].max()) if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
                efflux_cut = np.nanpercentile(all_efflux, 25)
            if influx_cut == 'auto':
                all_influx = [flux_norm(influx_mat.loc[m,c].max()) if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
                influx_cut = np.nanpercentile(all_influx, 25)
        else:
            comm_res[x1] = [efflux_mat.loc[m,c].max() if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
            comm_res[x2] = [influx_mat.loc[m,c].max() if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
            if efflux_cut == 'auto':
                all_efflux = [efflux_mat.loc[m,c].max() if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
                efflux_cut = np.nanpercentile(all_efflux, 25)
            if influx_cut == 'auto':
                all_influx = [influx_mat.loc[m,c].max() if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
                influx_cut = np.nanpercentile(all_influx, 25)        
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])]
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        update_commu_res = pd.concat([tmp1, tmp2, tmp_na])
        if inplace:
            self.efflux_mat = efflux_mat
            self.influx_mat = influx_mat
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)

    def _ConstainCompassFlux_(self, compass_folder, efflux_cut = 'auto', influx_cut='auto', inplace=True):
        """
        a function to filter out communications with low efflux and influx rates based on COMPASS output, the commu_res will be replaced by updated table
        Params
        -----
        compass_folder: a string for folder path or a dict {condition: path}. The path indicates COMPASS output folder. The output folder should include secretions.tsv and uptake.tsv for cell group level.
        efflux_cut: a numeric efflux threshold to indicate active efflux event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        influx_cut: a numeric ifflux threshold to indicate active influx event. Default sets to 'auto', which determines the threshold by taking 25th percentile of COMPASS values after square root transfermation ((x/np.abs(x)) * np.sqrt(np.abs(x)))
        inplace: True for updating the commu_res in the object, False for return the updated communication table without changing the mebo_obj
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        ## compass
        efflux_mat, influx_mat = FBA._get_compass_flux_(compass_folder=compass_folder, 
                                           compass_met_ann_path=_read_config(self.config_path)['common']['compass_met_ann_path'], 
                                                                  met_ann=self.met_ann)
        # self._get_compass_flux_(compass_folder = compass_folder)
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        comm_res[x1] = [efflux_mat.loc[m,c] if m in efflux_mat.index.tolist() else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res[x2] = [influx_mat.loc[m,c] if m in influx_mat.index.tolist() else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        comm_res[x1] = [flux_norm(x) for x in comm_res[x1].tolist()]
        comm_res[x2] = [flux_norm(x) for x in comm_res[x2].tolist()]
        if efflux_cut == 'auto':
            all_efflux = [flux_norm(efflux_mat.loc[m,c]) if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
            efflux_cut = np.nanpercentile(all_efflux, 25)
        if influx_cut == 'auto':
            all_influx = [flux_norm(influx_mat.loc[m,c]) if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
            influx_cut = np.nanpercentile(all_influx, 25)
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])]
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        update_commu_res = pd.concat([tmp1, tmp2, tmp_na])
        if inplace:
            self.efflux_mat = efflux_mat
            self.influx_mat = influx_mat
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)
            
    def _cobra_run_(self, gem_file, thread = 8, save_to_dir = '_tmp_cobra_fba_'):
        """
        perform COBRA fba analysis, flux result for each sample saved to a csv, named by ID csv files, e.g. 1.csv,
        return a dict with ID and sample (cell group) relationships for easily covert back 
        """
        avg_exp = pd.DataFrame(self.avg_exp.toarray(), index = self.avg_exp_indexer, columns = self.avg_exp_columns)

        return( FBA._cobra_run_(gem_file = gem_file, avg_exp = avg_exp, thread = thread, tmp_path = save_to_dir) )
    
    def _ConstrainCobraFlux_(self,
                        cobra_folder,
                        gem_path,
                        sample_id_dict,
                        efflux_cut = 'auto', 
                        influx_cut = 'auto',
                        inplace=True
                        ):
        """
        perform Flux-Balance Analysis for each cell group
        """
        comm_res = self.commu_res.sort_values(['Sender', 'Receiver', 'Metabolite', 'Sensor'])
        info('Collect Cobra flux result')
        cobra_efflux_dict, cobra_influx_dict = FBA._cobra_collect_(comm_res, cobra_folder, gem_path, self.met_ann, sample_id_dict)

        ## concate to commu_res
        x1 = 'sender_transport_flux'
        x2 = 'receiver_transport_flux'
        flux_norm = lambda x: (x/np.abs(x)) * np.sqrt(np.abs(x)) if x != 0 else 0
        comm_res[x1] = [flux_norm(efflux_mat.loc[m,c]) if m in cobra_efflux_dict else np.nan for c, m in comm_res[['Sender', 'Metabolite_Name']].values.tolist()]
        comm_res[x2] = [flux_norm(influx_mat.loc[m,c]) if m in cobra_influx_dict else np.nan for c, m in comm_res[['Receiver', 'Metabolite_Name']].values.tolist()]
        if efflux_cut == 'auto':
            all_efflux = [flux_norm(efflux_mat.loc[m,c]) if m in efflux_mat.index.tolist() else np.nan for c, m in self.original_result[['Sender', 'Metabolite_Name']].values.tolist()]
            efflux_cut = np.nanpercentile(all_efflux, 25)
        if influx_cut == 'auto':
            all_influx = [flux_norm(influx_mat.loc[m,c]) if m in influx_mat.index.tolist() else np.nan for c, m in self.original_result[['Receiver', 'Metabolite_Name']].values.tolist()]
            influx_cut = np.nanpercentile(all_influx, 25)
        print('efflux_cut:', efflux_cut)
        print('influx_cut:', influx_cut)
        ## base_efflux_influx_cut
        tmp_na = comm_res[pd.isna(comm_res[x1]) | pd.isna(comm_res[x2])]
        tmp1 = comm_res.query('Annotation != "Receptor"').copy()
        tmp2 = comm_res.query('Annotation == "Receptor"').copy()
        tmp1 = tmp1[(tmp1[x1]>efflux_cut) & (tmp1[x2]>influx_cut)]
        tmp2 = tmp2[(tmp2[x1]>efflux_cut)]
        update_commu_res = pd.concat([tmp1, tmp2, tmp_na])
        if inplace:
            self.efflux_mat = pd.DataFrame(cobra_efflux_dict).T
            self.influx_mat = pd.DataFrame(cobra_influx_dict).T
            self.commu_res = update_commu_res.copy()
        else:
            return(update_commu_res)


    def infer_commu(self, 
                      n_shuffle = 1000,
                      seed = 12345, 
                      Return = True, 
                      thread = None,
                      save_permuation = False,
                      pval_method='permutation_test_fdr',
                      pval_cutoff=0.05,
                      min_cell_number = 50
                     ):
        """
        execute mebocost to infer communications

        Params
        -----
        n_shuffle
            int, number of cell label shuffling for generating null distribution when calculating p-value
            
        seed
            int, a random seed for shuffling cell labels, set seed to get reproducable shuffling result 
            
        Return
            True or False, set True to return the communication event in a data frame
            
        thread
            int, the number of cores used in the computing, default None, thread set when create the object has the highest priority to be considered, so only set thread here if you want to make a change
            
        save_permuation
            True or False, set True to save the communication score for each permutation, this could occupy a higher amount of space when saving out, so default is False
        pval_method
            should be one of ['ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        min_cell_number
            int, the cell groups will be excluded and p-value will be replaced to 1 if there are not enough number of cells (less than min_cell_number), default is 50

        """
        tic = time.time()
        today = datetime.today().strftime("%B %d, %Y")
        now = datetime.now().strftime("%H:%M:%S")
        current_time = today + ' ' + now
        self.commu_time_stamp = current_time
        self.pval_method = pval_method
        self.pval_cutoff = pval_cutoff
        self.min_cell_number = min_cell_number
        tracemalloc.start()
        ## load config
        self._load_config_()
        
        ## take average by cell group, this must be done before extract sensor and enzyme gene expression of cells
        self._avg_by_group_()
        
        ## extract exp data for sensor and enzyme genes for all cells
        self._get_gene_exp_()
        
        ## estimate metabolite
        self.estimator()
        
        ## avg met mat
        self._avg_met_group_()
    
        # running communication inference
        commu_res_df, commu_res_bg = self.infer(
                                                n_shuffle = n_shuffle, 
                                                seed = seed,
                                                thread = self.thread if thread is None else thread ## allow to set thread in this function
                                                )
        ## update self
        self.commu_res = commu_res_df
        if save_permuation:
            self.commu_bg = commu_res_bg
        
        ## check cell proportion
        self._check_aboundance_()

        ## check low and set p val to 1
        commu_res_df_updated = self._filter_lowly_aboundant_(pvalue_res = commu_res_df,
                                                             cutoff_prop = self.cutoff_prop,
                                                             met_prop=self.met_prop, 
                                                             exp_prop=self.exp_prop,
                                                             pval_method='permutation_test_fdr',
                                                             pval_cutoff=0.05,
                                                             min_cell_number = min_cell_number)
        ## update self
        self.commu_res = commu_res_df_updated[(commu_res_df_updated[pval_method]<pval_cutoff)]
        
        current, peak = tracemalloc.get_traced_memory()
        
        # stopping the library
        tracemalloc.stop()
        
        toc = time.time()
        info('Prediction Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))
        if Return:
            return(commu_res_df_updated)
        
## ============================= differential communication analyisis =====================
    def CommDiff(self,
                comps=[],
                sig_mccc_only = True,
                thread = 8,
                Return=False
                ):
        """
        differential communication function

        Param:
        -----
        comps: a list, format should be cond1__vs__cond2, for example, Tumor__vs__Normal will indicate
                         to compare mCCC using Tumor vs. Normal.

        Return
        ----
        if Return set to True, a dict will be return to show the diff mCCC table in each comparison
        """
        tracemalloc.start()
        tic = time.time()
        
        diff_res = collections.defaultdict()
        for comp in comps:
            info('Diff Comm for {}'.format(comp))
            diff_res[comp] = CD.DiffComm(cell_ann=self.cell_ann,
                                        condition_col=self.condition_col,
                                        group_col=self.group_col,
                                        original_result=self.original_result,
                                        commu_res = self.commu_res.copy() if sig_mccc_only else pd.DataFrame(),
                                        commu_bg=self.commu_bg,
                                        prop_cut = self.cutoff_prop,
                                        comparison=comp, thread = thread)
        self.diffcomm_res = diff_res
        
        # stopping the library
        toc = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info('Diff mCCC Analysis Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))
        
        if Return:
            return(diff_res)

    def DiffFlowPlot(self,
                    comp_cond,
                    pval_method='permutation_test_fdr',
                    pval_cutoff=0.05,
                    Log2FC_threshold = 0,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    remove_unrelevant = False,
                    and_or = 'and',
                    node_label_size = 8,
                    node_alpha = .8,
                    figsize = 'auto',
                    node_cmap = 'Set1',
                    line_color_col = 'Log2FC',
                    line_cmap = 'coolwarm',
                    line_cmap_vmin = None,
                    line_cmap_vmax = None,
                    line_cmap_center = None,
                    linewidth = 1.5,
                    node_size_norm = (10, 150),
                    node_value_range = None,
                    save=None, 
                    save_plot = False, 
                    show_plot = True,
                    text_outline = False,
                    return_fig = False):
        """
        plot diff flow plot, the line color reflects log2 fold change, by default, only significant diff mCCC be plotted
        """
        comm_res = self.diffcomm_res[comp_cond]
        cond1, cond2 = comp_cond.split('__vs__')
        # comm_res = comm_res[(comm_res['Sig_'+cond1] == True) | (comm_res['Sig_'+cond2] == True)]
        
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
            
        fig = CP._DiffFlowPlot_(comm_res = comm_res, pval_method=pval_method, pval_cutoff=pval_cutoff, Log2FC_threshold=Log2FC_threshold,
                    sender_focus = sender_focus, metabolite_focus = metabolite_focus, sensor_focus = sensor_focus,
                    receiver_focus = receiver_focus, remove_unrelevant = remove_unrelevant, and_or = and_or,
                    node_label_size = node_label_size, node_alpha = node_alpha, figsize = figsize,
                    node_cmap = node_cmap, line_color_col = line_color_col, line_cmap = line_cmap, 
                    line_cmap_vmin = line_cmap_vmin, line_cmap_vmax = line_cmap_vmax, line_cmap_center = line_cmap_center,
                    linewidth = linewidth, node_size_norm = node_size_norm, node_value_range = node_value_range,
                    pdf=Pdf, save_plot = save_plot, show_plot = show_plot, text_outline = text_outline, return_fig = return_fig)
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
            
    def CompScatterPlot(self, comp_cond, 
                    pval_method = 'permutation_test_fdr',
                    pval_threshold = 0.05,
                    Log2FC_threshold = 0,
                    show_plot = True,
                    figsize = (5.5, 4),
                    return_fig = False,
                    save = None
                   ):
        """
        generate a scatter plot to show the difference
        """
        if comp_cond not in self.diffcomm_res:
            raise KeyError('%s is not in the comparision results!'%comp_cond)

        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
            
        cond1, cond2 = comp_cond.split('__vs__')
        comm_res = self.diffcomm_res[comp_cond]
        # comm_res = comm_res[(comm_res['Sig_'+cond1] == True) | (comm_res['Sig_'+cond2] == True)]

        fig = CP.CompScatterPlot(comm_res, cond1 = cond1, cond2 = cond2, pval_method = pval_method,
                    pval_threshold = pval_threshold, Log2FC_threshold = Log2FC_threshold, 
                    show_plot = show_plot, figsize = figsize,
                    return_fig = return_fig, save = Pdf, title = comp_cond)
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
            
# ## ============================== pathway association ============================
    def pathway_association(self,
                            adata=None,
                            exp_df=None,
                            meta_df=None,
                            comm_res = pd.DataFrame(),
                            cg_col = 'cell_group',
                            method = 'VISION',
                            pathway_db = ['KEGG', 'Wikipathway', 'Reactome'],
                            gmt_file=[],
                            num_workers = 8,
                            AUC_threshold = 0.05
                           ):
        """
        method: can be either VISION or AUCell
        """
        tracemalloc.start()
        tic = time.time()

        if adata is None and exp_df is None:
            raise KeyError('Please provide either adata or exp_df')
            
        if adata is None and exp_df is not None and meta_df is not None:
            adata = sc.AnnData(X = exp_df, obs = meta_df)
        
        if cg_col not in adata.obs.columns.tolist():
            raise KeyError('cg_col: %s is not in meta table [obs, meta_df]'%cg_col)

        if comm_res.shape[0] == 0:
            info('comm_res was empty, so trying to find significant mCCC table from mebocost object automatically')
            try:
                comm_res = self.commu_res.query('permutation_test_fdr < 0.05')
                if comm_res.shape[0] == 0:
                    raise ValueError('Failed to find significant comm_res in the object')
            except:
                raise ValueError('Failed to find comm_res in the object')

        gmt_path = [self.gmt_path[x] for x in pathway_db if x in self.gmt_path] + gmt_file
        print(gmt_path)
        
        if len(gmt_path) == 0:
            raise ValueError('No gmt files')

        if method == 'AUCell':
            self.aucs_pathy_ad, self.aucs_sensor_cnt_pathy, self.aucs_sensor_cnt_pathy_corr = PA.aucell_caculator(adata = adata, 
                                                                                             gmt_path=gmt_path,
                                                                                             commu_res = comm_res,
                                                                                             AUC_threshold = AUC_threshold,
                                                                                             num_workers = num_workers,
                                                                                             cg_col = cg_col)
        if method == 'VISION':
            self.vis_pathy_ad, self.vis_sensor_cnt_pathy, self.vis_sensor_cnt_pathy_corr = PA.vision_caculator(adata=adata, 
                                                                                         gmt_path=gmt_path,
                                                                                         commu_res = comm_res,
                                                                                         cg_col = cg_col)
        # stopping the library
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        toc = time.time()
        info('Pathway Association Done in {:.4f} seconds'.format(toc-tic))
        info('Memory Usage in Peak {:.2f} GB'.format(peak / 1024 / 1024 / 1024))

    def getSensorPathyCorrPathy(self, sensor, receiver, method = 'VISION'):
        # sensor = 'P2rx1'
        # receiver = 'VSM'
        if method == 'VISION':
            sensor_cont_pathway = self.vis_sensor_cnt_pathy
            sensor_cont_corr_pathway = self.vis_sensor_cnt_pathy_corr
            pathy_adata = self.vis_pathy_ad
        elif method == 'AUCell':
            sensor_cont_pathway = self.aucs_sensor_cnt_pathy
            sensor_cont_corr_pathway = self.aucs_sensor_cnt_pathy_corr
            pathy_adata = self.aucs_pathy_ad
        else:
            raise KeyError('method should be either AUCell or VISION')
            
        if sensor in sensor_cont_pathway:
            pathways = sensor_cont_pathway[sensor]
            if receiver not in sensor_cont_corr_pathway[sensor]:
                raise KeyError('Receiver was not found in the sensor containing pathway associations')
            corr_mat = {}
            for pathway in pathways:
                corr_mat[sensor+'__'+pathway] = pd.Series(sensor_cont_corr_pathway[sensor][receiver][pathway], index = pathy_adata.var_names)
            corr_mat = pd.DataFrame.from_dict(corr_mat)
            corr_mat['average_pathway_score'] = sensor_cont_corr_pathway[sensor][receiver]['average_pathway_activity']
            corr_mat.index.name = None
            corr_mat = corr_mat.sort_values(corr_mat.columns[0], ascending = False)
            return(corr_mat)
        else:
            info('Sensor was not found in sensor containing pathways')
            
    
    def SensorPathwayPlot(self, 
                          sensor,
                          method = 'VISION',
                          cg_col = 'cell_group',
                          receiver = [],
                          cg_focus = [],
                          save = None,
                          show_plot = True,
                          return_fig = False,
                          figsize = None):

         ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        
        if method == 'VISION':
            if sensor not in self.vis_sensor_cnt_pathy:
                raise KeyError('Sensor was not found in sensor containing pathways')
                
            fig = PA.SensorPathwayPlot(pathy_adata=self.vis_pathy_ad,
                     sensor_cont_pathway=self.vis_sensor_cnt_pathy,
                     sensor=sensor, cg_col = cg_col, receiver = receiver, cg_focus = cg_focus,
                     pdf = Pdf, show_plot = show_plot, return_fig = return_fig,
                     figsize = figsize)
            
            if fig is None and return_fig is True:
                raise ValueError('No fig data')

        elif method == "AUCell":
            if sensor not in self.aucs_sensor_cnt_pathy:
                raise KeyError('Sensor was not found in sensor containing pathways')
                
            fig = PA.SensorPathwayPlot(pathy_adata=self.aucs_pathy_ad,
                     sensor_cont_pathway=self.aucs_sensor_cnt_pathy,
                     sensor=sensor, cg_col = cg_col, receiver = receiver, cg_focus = cg_focus,
                     pdf = Pdf, show_plot = show_plot, return_fig = return_fig,
                     figsize = figsize)
                        
            if fig is None and return_fig is True:
                raise ValueError('No fig data')
                
        else:
            raise KeyError('method should be either AUCell or VISION')
            
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def PathwayScatterPlot(self, 
                          pathw_1,
                          pathw_2,
                          receiver,
                      method = 'VISION',
                      cg_col = 'cell_group',
                      save = None,
                      show_plot = True,
                      return_fig = False,
                      figsize = (4.5, 4)
                      ):
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        
        if method == 'VISION':
            
            
            fig = PA.PathwayScatterPlot(pathy_adata=self.vis_pathy_ad,
                      pathw_1=pathw_1, pathw_2=pathw_2,
                      receiver=receiver, cg_col = cg_col,
                      pdf = Pdf, show_plot = show_plot,
                      return_fig = return_fig, figsize = figsize)
            
            if fig is None and return_fig is True:
                raise ValueError('No fig data')

        elif method == "AUCell":
            
            fig = PA.PathwayScatterPlot(pathy_adata=self.aucs_pathy_ad,
                      pathw_1=pathw_1, pathw_2=pathw_2,
                      receiver=receiver, cg_col = cg_col,
                      pdf = Pdf, show_plot = show_plot,
                      return_fig = return_fig, figsize = figsize)
           
            if fig is None and return_fig is True:
                raise ValueError('No fig data')
                
        else:
            raise KeyError('method should be either AUCell or VISION')
            
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

                                                                    
# ## ============================== pathway inference ============================
#     def infer_pathway(self, 
#                      pval_method='permutation_test_fdr', 
#                      pval_cutoff = 0.05, 
#                      commu_score_cutoff = 0, 
#                      commu_score_column = 'Commu_Score', 
#                      min_term = 15, 
#                      max_term = 500,
#                      thread = None,
#                      sender_focus = [],
#                      metabolite_focus = [],
#                      sensor_focus = [],
#                      receiver_focus = [],
#                      Return_res = False, 
#                     ):
#         """
#         execute MEBOCOST to infer communication associated pathways
        
#         Param
#         ------
#         pval_method
#             should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        
#         pval_cutoff
#             float, a value in range between 0 and 1, pvalue less than the cutoff considered as significant event
            
#         commu_score_cutoff
#             float, communication score greater than the cutoff considered as a good event
            
#         commu_score_column
#             str, a column name in commu_res table, the column will be considered as communication score column and the communication score greater than the cutoff set by commu_score_cutoff considered as a good event

#         min_term
#             int, the pathway will be included when the number of genes in the pathway greater than min_term, default is 15
            
#         max_term
#             int, the pathway will be included when the number of genes in the pathway less than max_term, default is 500

        
#         thread 
#             int, the number of cores used in the computing, default None, thread set when create the object has the highest priority to be considered, so only set thread here if you want to make a change
        
#         sender_focus
#             a list of sender cell type or cell groups that will be focused in the analysis
            
#         metabolite_focus
#             a list of metabolite name that will be focused in the analysis
            
#         sensor_focus
#             a list of sensor name that will be focused in the analysis
            
#         receiver_focus
#             a list of receiver cell type or cell groups that will be focused in the analysis
        
#         Return_res
#             True or False, set True to return the pathway enrichment result after running this function. set False to indicate that do not return in this function but the result will be saved in MEBOCOST object, default is False
            
#         """
#         tic = time.time()
#         today = datetime.today().strftime("%B %d, %Y")
#         now = datetime.now().strftime("%H:%M:%S")
#         current_time = today + ' ' + now
#         self.pathway_time_stamp = current_time
        
#         ## check communication 
#         if 'commu_res' not in dir(self):
#             info('Communication result cannot be found, please excute _predict_() function first!')
#             return
        
#         ## good communications 
#         good_commu = self.commu_res[(self.commu_res[pval_method] < pval_cutoff) &
#                                     (self.commu_res[commu_score_column] > commu_score_cutoff)]
#         ## start a object                      
#         eobj = PE.PathwayEnrich(commu_res = good_commu,
#                                 gene_network=self.gene_network,
#                                 avg_exp = self.avg_exp,
#                                 avg_exp_indexer = self.avg_exp_indexer,
#                                 avg_exp_columns = self.avg_exp_columns,
#                                 cell_ann = self.cell_ann,
#                                 gmt_path = self.gmt_path,
#                                 min_term = min_term,
#                                 max_term = max_term,
#                                 thread = self.thread if thread is None else thread
#                                 )
        
#         self.enrich_result = eobj._pred_(pval_method = pval_method, 
#                             sensor_in_receiver = True, 
#                             sender_to_receiver = True,
#                             sender_focus = sender_focus,
#                             metabolite_focus = metabolite_focus,
#                             sensor_focus = sensor_focus,
#                             receiver_focus = receiver_focus,
#                             Return = True
#                            )
        
# #         ## add
# #         self.enrich_result = vars(eobj)
        
#         toc  = time.time()
#         info('Pathway Inference Done in {:.4f} seconds'.format(toc-tic))
#         if Return_res:
#             return self.enrich_result['sensor_res'], self.enrich_result['cellpair_res']
        

## ============================== communication plot functions ============================
    def eventnum_bar(self,
                    sender_focus = [],
                    metabolite_focus = [],
                    sensor_focus = [],
                    receiver_focus = [],
                    conditions = [],
                    and_or = 'and',
                    xorder = [],
                    pval_method = 'permutation_test_fdr',
                    pval_cutoff = 0.05,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    figsize = 'auto',
                    save = None,
                    show_plot = True,
                    show_num = True,
                    include = ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'],
                    group_by_cell = True,
                    colorcmap = 'tab20',
                    return_fig = False
                  ):
        """
        this function summarize the number of communication events
        
        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        xorder
            a list to order the x axis
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        show_num
            True or False, whether label y-axis value to the top of each bar
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to further filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        include
            a list, contains one or more elements from ['sender-receiver', 'sensor', 'metabolite', 'metabolite-sensor'], we try to summarize the number of communications grouping by the given elements, if return_fig set to be True, only provide one for each run.
        group_by_cell
            True or False, only effective for metabolite and sensor summary, True to further label number of communications in cell groups, False to do not do that
        colormap
            only effective when group_by_cell is True, should be a python camp str, default will be 'tab20', or can be a dict where keys are cell group, values are RGB readable color
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself.
            
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot
        
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None

        commu_res = self.commu_res.copy()
        if conditions:
            indices = commu_res['Condition'].isin(conditions)
            commu_res = commu_res.loc[indices,:]
            
        fig = CP._eventnum_bar_(commu_res = commu_res,
                    sender_focus = sender_focus,
                    metabolite_focus = metabolite_focus,
                    sensor_focus = sensor_focus,
                    receiver_focus = receiver_focus,
                    and_or = and_or,
                    xorder = xorder,
                    pval_method = pval_method,
                    pval_cutoff = pval_cutoff,
                    comm_score_col = comm_score_col,
                    comm_score_cutoff = comm_score_cutoff,
                    cutoff_prop = cutoff_prop,
                    figsize = figsize,
                    pdf = Pdf,
                    show_plot = show_plot,
                    show_num = show_num,
                    include = include,
                    group_by_cell = group_by_cell,
                    colorcmap = colorcmap,
                    return_fig = return_fig
                  )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
    
    def commu_dotmap(self,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                conditions = [],
                and_or = 'and',
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05, 
                figsize = 'auto',
                cmap = 'Reds',
                cmap_vmin = None,
                cmap_vmax = None,
                cellpair_order = [],
                met_sensor_order = [],
                dot_size_norm = (10, 150),
                save = None, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_range = None,
                comm_score_cutoff = None,
                cutoff_prop = None,
                swap_axis = False,
                return_fig = False):
        """
        commu_dotmap to show all significant communication events
        
        Params
        -----
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        cmap
            colormap for dot color, default is Reds
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        save
            str, the file name to save the figure
        show_plot
             True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = comm_res['Condition'].isin(conditions)
            comm_res = comm_res.loc[indices,:]
            
        fig = CP._commu_dotmap_(comm_res=comm_res, 
                     sender_focus = sender_focus,
                     metabolite_focus = metabolite_focus,
                     sensor_focus = sensor_focus,
                     receiver_focus = receiver_focus,
                     and_or = and_or,
                     pval_method=pval_method, 
                     pval_cutoff=pval_cutoff,
                     cmap_vmin = cmap_vmin,
                     cmap_vmax = cmap_vmax,
                     cellpair_order = cellpair_order,
                     met_sensor_order = met_sensor_order,
                     figsize = figsize, 
                     comm_score_col = comm_score_col,
                     comm_score_range = comm_score_range,
                     comm_score_cutoff = comm_score_cutoff,
                     cutoff_prop = cutoff_prop,
                     cmap = cmap,
                     dot_size_norm = dot_size_norm,
                     pdf = Pdf, 
                     show_plot = show_plot,
                     swap_axis = swap_axis,
                     return_fig = return_fig
                    )
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
        
    def FlowPlot(self, 
                pval_method='permutation_test_fdr',
                pval_cutoff=0.05,
                sender_focus = [],
                metabolite_focus = [],
                sensor_focus = [],
                receiver_focus = [],
                conditions = [],
                remove_unrelevant = True,
                and_or = 'and',
                node_label_size = 8,
                node_alpha = .8,
                figsize = 'auto',
                node_cmap = 'Set1',
                line_cmap = 'bwr',
                line_cmap_vmin = None,
                line_cmap_vmax = None,
                linewidth_norm = (0.1, 1),
                linewidth_value_range = None,
                node_size_norm = (10, 150),
                node_value_range = None,
                save=None, 
                show_plot = True,
                comm_score_col = 'Commu_Score',
                comm_score_cutoff = None,
                cutoff_prop = None,
                text_outline = False,
                return_fig = False):
        """
        Flow plot to show the communication connections from sender to metabolite, to sensor, to receiver

        Params
        ------
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        remove_unrelevant
            True or False, set True to hide unrelated nodes 
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        node_label_size
            float, font size of text label on node, default will be 8
        node_alpha
            float, set to transparent node color
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        node_cmap
            node color map or a four-element list, used to color sender, metabolite, sensor, receiver, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_cmap
            line color map, used to indicate the communication score, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        node_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        linewidth_norm
            two values in a tuple, used to normalize the line width, such as (0.1, 1)
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = comm_res['Condition'].isin(conditions)
            comm_res = comm_res.loc[indices,:]
        fig = CP._FlowPlot_(comm_res=comm_res, pval_method=pval_method, pval_cutoff=pval_cutoff, 
                      sender_focus = sender_focus, metabolite_focus = metabolite_focus,
                      sensor_focus = sensor_focus, receiver_focus = receiver_focus, 
                      remove_unrelevant = remove_unrelevant, and_or = and_or,
                      node_label_size = node_label_size, node_alpha = node_alpha, figsize = figsize, 
                      node_cmap = node_cmap, line_cmap = line_cmap, line_cmap_vmin = line_cmap_vmin,
                      line_cmap_vmax = line_cmap_vmax, linewidth_norm = linewidth_norm, 
                      linewidth_value_range = linewidth_value_range, node_value_range = node_value_range,
                      node_size_norm = node_size_norm, pdf=Pdf, show_plot = show_plot, 
                      comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                      text_outline = text_outline, return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)
    def count_dot_plot(self, 
                    conditions = [],
                    pval_method='permutation_test_pval', 
                    pval_cutoff=0.05, 
                    cmap='RdBu_r', 
                    figsize = 'auto',
                    save = None,
                    dot_size_norm = (5, 100),
                    dot_value_range = None,
                    dot_color_vmin = None,
                    dot_color_vmax = None,
                    show_plot = True,
                    comm_score_col = 'Commu_Score',
                    comm_score_cutoff = None,
                    cutoff_prop = None,
                    dendrogram_cluster = True,
                    sender_order = [],
                    receiver_order = [],
                    return_fig = False):
        """
        dot plot to show the summary of communication numbers between sender and receiver 

        Params
        -----
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        cmap
            color map to set dot color 
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        save
            str, the file name to save the figure
        dot_size_norm
            two values in a tuple, used to normalize the dot size, such as (10, 150)
        dot_color_vmin
            float, the value limits the color map in maximum
        dot_color_vmax
            float, the value limits the color map in minimum
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        
#         if show_plot is None and self.show_plot is not None:
#             show_plot = self.show_plot

        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = comm_res['Condition'].isin(conditions)
            comm_res = comm_res.loc[indices,:]
        fig = CP._count_dot_plot_(commu_res=comm_res, pval_method = pval_method, pval_cutoff = pval_cutoff, 
                        cmap = cmap, figsize = figsize, pdf = Pdf, dot_size_norm = dot_size_norm, dot_value_range = dot_value_range,
                        dot_color_vmin = dot_color_vmin, dot_color_vmax = dot_color_vmax, show_plot = show_plot,
                        comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                        dendrogram_cluster = dendrogram_cluster,
                        sender_order = sender_order, receiver_order = receiver_order,
                        return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        if return_fig:
            return(fig)

    def commu_network_plot(self,
                        sender_focus = [],
                        metabolite_focus = [],
                        sensor_focus = [],
                        receiver_focus = [],
                        conditions = [],
                        remove_unrelevant = False,
                        and_or = 'and',
                        pval_method = 'permutation_test_fdr',
                        pval_cutoff = 0.05,
                        node_cmap = 'tab20',
                        figsize = 'auto',
                        line_cmap = 'RdBu_r',
                        line_color_vmin = None,
                        line_color_vmax = None,
                        linewidth_value_range = None,
                        linewidth_norm = (0.1, 1),
                        node_size_norm = (50, 300),
                        node_value_range = None,
                        adjust_text_pos_node = True,
                        node_text_hidden = False,
                        node_text_font = 10,
                        save = None,
                        show_plot = True,
                        comm_score_col = 'Commu_Score',
                        comm_score_cutoff = None,
                        cutoff_prop = None,
                        text_outline = False,
                        return_fig = False):

        """
        Network plot to show the communications between cell groups

        Params
        ------
        sender_focus
            a list, set a list of sender cells to be focused, only plot related communications
        metabolite_focus
            a list, set a list of metabolites to be focused, only plot related communications
        sensor_focus
            a list, set a list of sensors to be focused, only plot related communications
        receiver_focus
            a list, set a list of receiver cells to be focused, only plot related communications
        remove_unrelevant
            True or False, set True to hide unrelated nodes
        and_or
            eithor 'and' or 'or', 'and' for finding communications that meet to all focus, 'or' for union
        pval_method
            should be one of ['zztest_pval', 'ttest_pval', 'ranksum_test_pval', 'permutation_test_pval', 'zztest_fdr', 'ttest_fdr', 'ranksum_test_fdr', 'permutation_test_fdr'], default is permutation_test_fdr
        pval_cutoff
            float, set to filter out non-significant communication events
        node_cmap
            node color map, used to indicate different cell groups, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        line_cmap
            line color map, used to indicate number of communication events, set one from https://matplotlib.org/stable/tutorials/colors/colormaps.html
        line_color_vmin
            float, the value limits the line color map in minimum
        line_color_vmax
            float, the value limits the line color map in maximum
        linewidth_norm
            two values in a tuple, used to normalize the dot size, such as (0.1, 1)
        node_size_norm
            two values in a tuple, used to normalize the node size, such as (50, 300)
        adjust_text_pos_node 
            True or Flase, whether adjust the text position to avoid overlapping automatically
        node_text_font
            float, font size for node text annotaion
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        cutoff_prop
            a float between 0 and 1, set a cutoff to filter out lowly abundant cell populations by the fraction of cells expressed sensor genes or metabolite, Note that this parameter will lost the function if cutoff_prop was set lower than the one user set at begaining of running mebocost.infer_commu or preparing mebocost object. This parameter were designed to further strengthen the filtering.
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        
        comm_res = self.commu_res
        ## pdf
        if save is not None and save is not False and isinstance(save, str):
            Pdf = PdfPages(save)
        else:
            Pdf = None
        if conditions:
            indices = comm_res['Condition'].isin(conditions)
            comm_res = comm_res.loc[indices,:]
        fig = CP._commu_network_plot_(commu_res=comm_res, sender_focus = sender_focus, metabolite_focus = metabolite_focus, 
                            sensor_focus = sensor_focus, receiver_focus = receiver_focus, and_or = and_or, 
                            pval_method = pval_method, remove_unrelevant = remove_unrelevant,
                            pval_cutoff = pval_cutoff, node_cmap = node_cmap, figsize = figsize, line_cmap = line_cmap, 
                            line_color_vmin = line_color_vmin, line_color_vmax = line_color_vmax,
                            linewidth_norm = linewidth_norm, linewidth_value_range = linewidth_value_range, node_text_hidden = node_text_hidden,
                            node_size_norm = node_size_norm, node_value_range = node_value_range, adjust_text_pos_node = adjust_text_pos_node, 
                            comm_score_col = comm_score_col, comm_score_cutoff = comm_score_cutoff, cutoff_prop = cutoff_prop,
                            node_text_font = node_text_font, pdf = Pdf, show_plot = show_plot, text_outline = text_outline,
                            return_fig = return_fig)
        if save is not None and save is not False and isinstance(save, str):
            Pdf.close()
        
        if return_fig:
            return(fig)
            
    def violin_plot(self,
                    sensor_or_met,
                    cell_focus = [],
                    cell_order = [],
                    conditions = [],
                    row_zscore = False,
                    cmap = None,
                    vmin = None,
                    vmax = None,
                    figsize = 'auto',
                    cbar_title = '',
                    save = None,
                    show_plot = True,
                    return_fig = False):
        """
        Violin plot to show the distribution of sensor expression or metabolite enzyme expression across cell groups

        Params
        -----
        sensor_or_met
            a list, provide a list of sensor gene name or metabolite name
        cell_focus
            a list, provide a list of cell type that you want to focus, otherwise keep empty
        cmap
            the color map used to draw the violin
        vmin
            float, maximum value for the color map
        vmin
            float, minimum value for the color map
        figsize
            auto or a tuple of float such as (5.5, 4.2), defualt will be automatically estimate
        title
            str, figure title on the top
        save
            str, the file name to save the figure
        show_plot
            True or False, whether print the figure on the screen
        comm_score_col
            column name of communication score, can be Commu_Score
        comm_score_cutoff
            a float, set a cutoff so only communications with score greater than the cutoff will be focused
        return_fig:
            True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
        """
        ## cell group
        cell_ann = self.cell_ann.copy()
        if conditions:
            cell_ann = cell_ann.loc[cell_ann[self.condition_col].isin(conditions),:]
            
        if 'cell_group' not in cell_ann.columns.tolist():
            raise ValueError('ERROR: "cell_group" not in cell_ann column names!')
        ### extract expression for sensor
        sensors = []
        if self.exp_mat is not None and self.exp_mat_indexer is not None:
            sensor_loc = np.where(pd.Series(self.exp_mat_indexer).isin(sensor_or_met))
            #[i for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            sensors = self.exp_mat_indexer[sensor_loc]
            #[j for i,j in enumerate(self.exp_mat_indexer.tolist()) if j in sensor_or_met]
            exp_dat = pd.DataFrame(self.exp_mat[sensor_loc].toarray(),
                                   index = sensors,
                                   columns = self.exp_mat_columns)
            
            if len(sensors) > 0:
                info('Find genes %s to plot violin'%(sensors))
                ## expression
                if save is not None and save is not False and isinstance(save, str):
                    save = save.replace('.pdf', '_sensor_exp.pdf')
                    Pdf = PdfPages(save)
                else:
                    Pdf = None
                if cmap is None:
                    ccmap = 'Reds'
                else:
                    ccmap = cmap

                if cbar_title == '':
                    if row_zscore:
                        sensor_cbar_title = 'Mean Z score of sensor expression'
                    else:
                        sensor_cbar_title = 'Mean sensor expression'
                else:
                    sensor_cbar_title = cbar_title
                ## data mat for plot
                dat_mat = pd.merge(exp_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True).dropna()
                fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(sensors),
                                       cell_focus = cell_focus, cell_order = cell_order, 
                                       cmap = ccmap, row_zscore = row_zscore,
                                       vmin = vmin, vmax = vmax, figsize = figsize, 
                                       cbar_title = sensor_cbar_title, pdf = Pdf,
                                       show_plot = show_plot, return_fig = return_fig)

                if save is not None and save is not False and isinstance(save, str):
                    Pdf.close()
                if return_fig:
                    return(fig)
            else:
                info('Warnings: no sensors to plot')
        else:
            info('Warnings: failed to load metabolite data matrix')
            
        ### extract metabolite level
        metabolites = list(set(sensor_or_met) - set(sensors))
        metabolites = list(set(metabolites) & set(self.met_ann['metabolite'].unique().tolist()))
        if metabolites:
            # to HMDBID
            met_name_to_id = {}
            for m, iD in self.met_ann[['metabolite', 'HMDB_ID']].values.tolist():
                met_name_to_id[m] = iD
            metaboliteIds = {x: met_name_to_id.get(x) for x in metabolites}
            ## metabolite matrix
            if self.met_mat is not None and self.met_mat_indexer is not None:
                met_loc = np.where(pd.Series(self.met_mat_indexer).isin(list(metaboliteIds.values())))[0]
                met_Ids = self.met_mat_indexer[met_loc]
                met_names = [list(metaboliteIds.keys())[list(metaboliteIds.values()).index(x)] for x in met_Ids]
                met_dat = pd.DataFrame(self.met_mat[met_loc].toarray(),
                                   index = met_names,
                                   columns = self.met_mat_columns)
                dat_mat = pd.merge(met_dat.T, cell_ann[['cell_group']], left_index = True, right_index = True).dropna()
                if len(met_names) > 0:
                    info("Find metabolites %s to plot violin"%metabolites)
                    ## expression
                    if save is not None and save is not False and isinstance(save, str):
                        save = save.replace('.pdf', '_metabolite.pdf')
                        Pdf = PdfPages(save)
                    else:
                        Pdf = None
                    if cmap is None:
                        ccmap = 'Purples'
                    else:
                        ccmap = cmap
                    if cbar_title == '':
                        if row_zscore:
                            met_cbar_title = 'Mean Z score of\n aggregated enzyme expression'
                        else:
                            met_cbar_title = 'Mean aggregated enzyme expression'
                    else:
                        met_cbar_title = cbar_title
                        
                    fig = CP._violin_plot_(dat_mat=dat_mat, sensor_or_met=list(metaboliteIds.keys()),
                                     cell_focus = cell_focus, cmap = ccmap,
                                     cell_order = cell_order, row_zscore = row_zscore, 
                                    vmin = vmin, vmax = vmax, figsize = figsize,
                                    cbar_title = met_cbar_title, pdf = Pdf,
                                    show_plot = show_plot, return_fig = return_fig)

                    if save is not None and save is not False and isinstance(save, str):
                        Pdf.close()
                    if return_fig:
                        return(fig)
                else:
                    info('Warnings: no metabolites to plot')
            else:
                info('Warnings: failed to load metabolite data matrix')
        else:
            info('Warnings: no metabolites to plot')

# ============ notebook ==========
    def communication_in_notebook(self,
                                  pval_method = 'permutation_test_fdr',
                                  pval_cutoff = 0.05,
                                  comm_score_col = 'Commu_Score',
                                  comm_score_cutoff = None, 
                                  cutoff_prop = None
                                 ):

        # some handy functions to use along widgets
        from IPython.display import display, Markdown, clear_output, HTML
        import ipywidgets as widgets
        import functools

        outt = widgets.Output()

        df = self.commu_res.copy()
        
        if not comm_score_cutoff:
            comm_score_cutoff = 0
        if not cutoff_prop:
            cutoff_prop = 0
        ## basic filter
        df = df[(df[pval_method] <= pval_cutoff) & 
                (df[comm_score_col] >= comm_score_cutoff) &
                (df['metabolite_prop_in_sender'] >= cutoff_prop) &
                (df['sensor_prop_in_receiver'] >= cutoff_prop)
                ]
        
        senders = ['All']+sorted(list(df['Sender'].unique()))
        receivers = ['All']+sorted(list(df['Receiver'].unique()))
        metabolites = ['All']+sorted(list(df['Metabolite_Name'].unique()))
        transporters = ['All']+sorted(list(df['Sensor'].unique()))
        
        logic_butt = widgets.RadioButtons(
                            options=['and', 'or'],
                            description='Logic',
                            disabled=False
                        )

        sender_sel = widgets.SelectMultiple(description='Sender:',
                                            options=senders,
                                            layout=widgets.Layout(width='30%'))
        receiver_sel = widgets.SelectMultiple(description='Receiver:',
                                              options=receivers,
                                              layout=widgets.Layout(width='30%'))
        metabolite_sel = widgets.SelectMultiple(description='Metabolite:',
                                                options=metabolites,
                                                layout=widgets.Layout(width='30%'))
        sensor_sel = widgets.SelectMultiple(description='Sensor:',
                                                 options=transporters,
                                                layout=widgets.Layout(width='30%'))
        
        flux_butt = widgets.Button(description='Communication Flow (FlowPlot)',
                              layout=widgets.Layout(width='100%'))
        net_butt = widgets.Button(description='Communication Network (CirclePlot)',
                              layout=widgets.Layout(width='100%'))
        dotHeatmap_butt = widgets.Button(description='Communication Details (Dot-shaped Heatmap)',
                              layout=widgets.Layout(width='100%'))
        violin_butt = widgets.Button(description='ViolinPlot to show metabolite or sensor level in cell groups',
                              layout=widgets.Layout(width='100%'))

        def _flowplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
        
                self.FlowPlot(pval_method=pval_method,
                            pval_cutoff=pval_cutoff,
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            remove_unrelevant = True,
                            and_or = and_or,
                            node_label_size = 8,
                            node_alpha = .8,
                            figsize = 'auto',
                            node_cmap = 'Set1',
                            line_cmap = 'bwr',
                            line_vmin = None,
                            line_vmax = None,
                            node_size_norm = (10, 150),
                            linewidth_norm = (0.5, 5),
                            save=None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop,
                            text_outline = False,
                            return_fig = False)
                
                
        def _networkplot_filter_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_network_plot(
                                sender_focus = [x for x in sender_sel.value if x != 'All'],
                                metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                                sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                                receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                                remove_unrelevant = False,
                                and_or = and_or,
                                pval_method = pval_method,
                                pval_cutoff = pval_cutoff,
                                node_cmap = 'tab20',
                                figsize = 'auto',
                                line_cmap = 'RdBu_r',
                                line_color_vmin = None,
                                line_color_vmax = None,
                                linewidth_norm = (0.1, 1),
                                node_size_norm = (50, 300),
                                adjust_text_pos_node = False,
                                node_text_font = 10,
                                save = None,
                                show_plot = True,
                                comm_score_col = comm_score_col,
                                comm_score_cutoff = comm_score_cutoff,
                                cutoff_prop = cutoff_prop,
                                text_outline = False
                                )
        def _dotHeatmapPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                and_or = logic_butt.value
                self.commu_dotmap(
                            sender_focus = [x for x in sender_sel.value if x != 'All'],
                            metabolite_focus = [x for x in metabolite_sel.value if x != 'All'],
                            sensor_focus = [x for x in sensor_sel.value if x != 'All'],
                            receiver_focus = [x for x in receiver_sel.value if x != 'All'],
                            and_or = and_or,
                            pval_method=pval_method,
                            pval_cutoff=pval_cutoff, 
                            figsize = 'auto',
                            cmap = 'bwr',
                            node_size_norm = (10, 150),
                            save = None, 
                            show_plot = True,
                            comm_score_col = comm_score_col,
                            comm_score_cutoff = comm_score_cutoff,
                            cutoff_prop = cutoff_prop
                )

        def _violinPlot_(b):
            with outt:
                clear_output()
                print('+++++++++++++++++++++++++++ Running, Please Wait +++++++++++++++++++++++++++ ')
                print('[Selection]: Sender{}; Metabolite{}; Transporter{}; Receiver{}'.format(sender_sel.value,
                                                                                                  metabolite_sel.value,
                                                                                                  sensor_sel.value,
                                                                                                  receiver_sel.value))
                
                self.violin_plot(
                                sensor_or_met = [x for x in metabolite_sel.value + sensor_sel.value if x != 'All'],
                                cell_focus = [x for x in sender_sel.value + receiver_sel.value if x != 'All'],
                                cmap = None,
                                vmin = None,
                                vmax = None,
                                figsize = 'auto',
                                cbar_title = '',
                                save = None,
                                show_plot = True)
                
                
        flux_butt.on_click(_flowplot_filter_)
        net_butt.on_click(_networkplot_filter_)
        dotHeatmap_butt.on_click(_dotHeatmapPlot_)
        violin_butt.on_click(_violinPlot_)


        h1 = widgets.HBox([sender_sel, metabolite_sel, sensor_sel, receiver_sel])
        h2 = widgets.VBox([flux_butt, net_butt, dotHeatmap_butt, violin_butt])

        mk = Markdown("""<b>Select and Click button to visulize</b>""")
        display(mk, widgets.VBox([logic_butt, h1, h2, outt]))

# ## ===================== pathway plot functions =============
        
#     def pathway_scatter(self, 
#                 a_pair, 
#                 pval_cutoff=0.05, 
#                 ES_cutoff=0,
#                 cmap = 'cool',
#                 vmax = None,
#                 vmin = None,
#                 figsize = 'auto',
#                 title = '',
#                 maxSize = 500,
#                 minSize = 15,
#                 save = None,
#                 show_plot = True,
#                 return_fig = False):
#         """
#         Plot the associated pathway in scatter plot
        
#         Params
#         -----
#         a_pair
#             str, the format should be either sensor ~ receiver or sender ~ receiver
#         pval_cutoff
#             float, cutoff of pval (padj) to focus on significantly associated pathways
#         ES_cutoff
#             float, cutoff of Fold Enrichment Score for assiciated pathways, positive value for positively associated pathways
#         cmap
#             python color map for showing significance
#         vmax
#             the maximum limits for colormap
#         vmin
#             the minimum limits for colormap
#         figsize
#             a tuple inclues two values, represents width and height, set "auto" to automatically estimate
#         title:
#             str, figure title
#         maxSize
#             the term size in maximum
#         minSize
#             the term size in mimum
#         save
#             str, the path of where the figure save to
#         show_plot
#             True or False, to display the figure on the screen or not
#         return_fig:
#             True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
#         """
        
#         cellpairs = list(self.enrich_result['cellpair_res'].keys())
#         sensor_receivers = list(self.enrich_result['sensor_res'].keys())

#         if a_pair in cellpairs:
#             res_dict = self.enrich_result['cellpair_res']
#         if a_pair in sensor_receivers:
#             res_dict = self.enrich_result['sensor_res']
#         if a_pair not in cellpairs and a_pair not in sensor_receivers:
#             raise KeyError('ERROR to read given a_pair!')
        
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf = PdfPages(save)
#         else:
#             Pdf = None

#         fig = PP._scatter_(res_dict = res_dict, a_pair = a_pair, pval_cutoff=pval_cutoff, 
#                     ES_cutoff=ES_cutoff, cmap = cmap, vmax = vmax, vmin = vmin,
#                     figsize = figsize, title = title, maxSize = maxSize, minSize = minSize,
#                     pdf = Pdf, show_plot = show_plot, return_fig = return_fig)
        
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf.close()
#         if return_fig:
#             return(fig)
        
#     def pathway_stacked_bar(self,
#                 pair1,
#                 pair2,
#                 pval_cutoff=0.05, 
#                 ES_cutoff=0,
#                 cmap = 'spring_r',
#                 vmax = None,
#                 vmin = None,
#                 figsize = 'auto',
#                 title = '',
#                 maxSize = 500,
#                 minSize = 15,
#                 colors = ['#CC6677', '#1E90FF'],
#                 save = None,
#                 show_plot = True,
#                 return_fig = False):
    
#         """
#         compare pathways for two communication events
        
#         Params
#         ------
#         pair1 and pair2
#             str, the format of pair1 and pairs should be both sensor ~ receiver, or both sender ~ receiver
#         pval_cutoff
#             float, cutoff of pval (padj) to focus on significantly associated pathways
#         ES_cutoff
#             float, cutoff of Fold Enrichment Score for assiciated pathways, positive value for positively associated pathways
#         cmap
#             python color map for showing significance
#         vmax
#             the maximum limits for colormap
#         vmin
#             the minimum limits for colormap
#         figsize
#             a tuple inclues two values, represents width and height, set "auto" to automatically estimate
#         title:
#             str, figure title
#         maxSize
#             the term size in maximum
#         minSize
#             the term size in mimum
#         save
#             str, the path of where the figure save to
#         show_plot
#             True or False, to display the figure on the screen or not
#         return_fig
#             True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
#         """
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf = PdfPages(save)
#         else:
#             Pdf = None
        
#         cellpairs = list(self.enrich_result['cellpair_res'].keys())
#         sensor_receivers = list(self.enrich_result['sensor_res'].keys())

#         cellpair = np.all([pair1 in cellpairs, pair2 in cellpairs])
#         sensor_receiver = np.all([pair1 in sensor_receivers, pair2 in sensor_receivers])

#         if cellpair:
#             res_dict = self.enrich_result['cellpair_res']
#         elif sensor_receiver:
#             res_dict = self.enrich_result['sensor_res']
#         else:
#             raise KeyError('pair1 and pair2 should be both sender-receiver or sensor-receiver, please check!')
        
            
#         fig = PP._stacked_bar_(res_dict = res_dict, pair1 = pair1, pair2 = pair2, 
#                          pval_cutoff = pval_cutoff, ES_cutoff = ES_cutoff,
#                          cmap = cmap, vmax = vmax, vmin = vmin, figsize = figsize,
#                          title = title, maxSize = maxSize, minSize = minSize, colors = colors,
#                          pdf = Pdf, show_plot = show_plot, return_fig = return_fig)
                
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf.close()
#         if return_fig:
#             return(fig)
            
    
#     def pathway_multi_dot(self,
#                     pairs, 
#                     pval_cutoff=0.05, 
#                     ES_cutoff=0,
#                     cmap = 'Spectral_r',
#                     vmax = None,
#                     vmin = None,
#                     node_size_norm = (20, 100),
#                     figsize = 'auto',
#                     title = '',
#                     maxSize = 500,
#                     minSize = 15,
#                     save = None,
#                     show_plot = True,
#                     swap_axis = False,
#                     return_fig = False):
#         """
#         draw dot map to show the associated pathways in multiple comparisons
        
#         Params
#         -----
#         pairs
#             a list, elements should be all in sender-receiver or all in sensor-receiver
#         pval_cutoff
#             float, cutoff of pval (padj) to focus on significantly associated pathways
#         ES_cutoff
#             float, cutoff of Normalized Enrichment Score for assiciated pathways, positive value for positively associated pathways, negative value for negatively associated
#         cmap
#             python color map for showing significance
#         vmax
#             the maximum limits for colormap
#         vmin
#             the minimum limits for colormap
#         figsize
#             a tuple inclues two values, represents width and height, set "auto" to automatically estimate
#         title:
#             str, figure title
#         maxSize
#             the term size in maximum
#         minSize
#             the term size in mimum
#         save
#             str, the path of where the figure save to
#         show_plot
#             True or False, to display the figure on the screen or not
#         swap_axis
#            True or False, set True to flip x and y axis
#         return_fig
#             True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself
#         """
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf = PdfPages(save)
#         else:
#             Pdf = None
        
#         cellpairs = list(self.enrich_result['cellpair_res'].keys())
#         sensor_receivers = list(self.enrich_result['sensor_res'].keys())
        
#         if np.all([x in cellpairs for x in pairs]):
#             res_dict = self.enrich_result['cellpair_res']
#         elif np.all([x in sensor_receivers for x in pairs]):
#             res_dict = self.enrich_result['sensor_res']
#         else:
#             raise KeyError('pairs should be a list of elements all in sender-receiver or all in sensor-receiver, please check!')

#         fig = PP._multi_dot_(res_dict = res_dict, pairs = pairs, pval_cutoff=pval_cutoff, ES_cutoff=ES_cutoff,
#                     cmap = cmap, vmax = vmax, vmin = vmin, node_size_norm = node_size_norm,
#                     figsize = figsize, title = title, maxSize = maxSize, minSize = minSize, pdf = Pdf,
#                     show_plot = show_plot, swap_axis = swap_axis, return_fig = return_fig)
        
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf.close()    
#         if return_fig:
#             return(fig)
        
#     def pathway_ES_plot(self, 
#                   a_pair, 
#                   pathway,
#                   figsize = (8, 3.5),
#                   dot_color = '#1874CD',
#                   curve_color = 'black',
#                   title='',
#                   save = None,
#                   show_plot = True,
#                   return_fig = False,
#                   return_data = False
#                  ):
#         """
#         this function to draw enrichment plot for certain pathway in certain cell
        
#         Params
#         ------
#         a_pair
#             str, the format should be either sensor ~ receiver or sender ~ receiver
#         pathway
#             str, pathway name
#         figsize
#             a tuple with two values, use to set the figure size, default is (10, 5)
#         dot_color
#             color name, use to set the pathway genes in the scatter plot
#         line_color
#             color name, use to set the pathway genes in enrichement plot
#         ht_cmap
#             color map, use to set the color scale for background
#         title
#             str, figure title
#         save
#             path, save the figure to
#         show_plot
#             True or False, display the figure on the screen or not
#         return_fig
#             True or False, set True to return the figure object, this can be useful if you want to manipulate figure by yourself        
#         """
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf = PdfPages(save)
#         else:
#             Pdf = None
        
#         ## geneset
#         all_pathway = self.enrich_result['gmt']
        
#         ## remove kegg id
#         if not re.search('hsa[0-9]* |mmu[0-9]* ', pathway):
#             all_pathway_update = {k.replace(re.search('hsa[0-9]* |mmu[0-9]* ', k).group(), '') : all_pathway[k] for k in all_pathway.keys()}
#             name_match = {k.replace(re.search('hsa[0-9]* |mmu[0-9]* ', k).group(), '') : k for k in all_pathway.keys()}
#         if pathway not in list(all_pathway_update.keys()):
#             raise KeyError('ERROR: cannot find the pathway in database, please check the name!')
#         geneSet = all_pathway_update[pathway]
#         pathway = name_match[pathway]
#         ##
#         cellpairs = list(self.enrich_result['cellpair_res'].keys())
#         sensor_receivers = list(self.enrich_result['sensor_res'].keys())

#         ## sensor in receiver
#         if a_pair in sensor_receivers:
#             s, r = a_pair.split(' ~ ')
#             s_loc = np.where(self.enrich_result['weighted_exp'][s]['index'] == r)
#             weightList = pd.Series(self.enrich_result['weighted_exp'][s]['weight_exp'][s_loc].toarray()[0],
#                                    index = self.enrich_result['weighted_exp'][s]['columns'])
#             expList = self.enrich_result['avg_exp_norm'][r]
#             gxtList = self.gene_network[s]
#             mHG_obj = self.enrich_result['sensor_res'][a_pair]['mHG_obj'][pathway]
            
#             fig = PP._sensor_ES_plot_(geneSet=geneSet, 
#                           mHG_obj=mHG_obj,
#                           expList=expList,
#                           gxtList=gxtList,
#                           sensor = s,
#                           receiver = r,
#                           figsize = figsize,
#                           dot_color = dot_color,
#                           curve_color = curve_color,
#                           title = title,
#                           pdf = Pdf,
#                           show_plot = show_plot,
#                           return_fig = return_fig
#                      )

#         ## sender to receiver
#         if a_pair in cellpairs:
#             sender, receiver = a_pair.split(' ~ ')
#             weightList = self.enrich_result['weighted_exp_agg'][a_pair]
#             expList = self.enrich_result['avg_exp_norm'][receiver]
#             mHG_obj = self.enrich_result['cellpair_res'][a_pair]['mHG_obj'][pathway]

#             fig = PP._cellpair_ES_plot_(geneSet=geneSet, 
#                       mHG_obj=mHG_obj,
#                       expList=expList,
#                       gxtList=weightList,
#                       sender = sender,
#                       receiver = receiver,
#                       figsize = figsize,
#                       dot_color = dot_color,
#                       curve_color = curve_color,
#                       title = title,
#                       pdf = Pdf,
#                       show_plot = show_plot,
#                       return_fig = return_fig
#                  )
            
                
#         if a_pair not in cellpairs and a_pair not in sensor_receivers:
#             raise KeyError('ERROR to read given a_pair!')
        
#         if save is not None and save is not False and isinstance(save, str):
#             Pdf.close()    
        
#         if return_data and return_fig:
#             df = pd.concat([expList, gxtList], axis = 1).reset_index().dropna()
#             df.columns = ['gene', 'scaled_expression', 'gene_correlation']
#             df = df.loc[df['gene'].isin(geneSet)]
#             return(fig, df)
#         elif return_data and not return_fig:
#             df = pd.concat([expList, weightList], axis = 1).reset_index().dropna()
#             df.columns = ['gene', 'scaled_expression', 'gene_weight']
#             df = df.loc[df['gene'].isin(geneSet)]
#             return(df)
#         elif return_fig and not return_data:
#             return(fig)
#         else:
#             pass
     
#     def pathway_in_notebook(self):

#         # some handy functions to use along widgets
#         from IPython.display import display, Markdown, clear_output, HTML
#         # widget packages
#         import ipywidgets as widgets
#         import functools

        
#         ## sensor ~ cell
#         sender_receiver = list(self.enrich_result['sensor_res'].keys())
#         sender_receiver = sorted(list(set(sender_receiver)))

#         ## cell -> cell
#         cellpair = list(self.enrich_result['cellpair_res'].keys())
#         cellpair = sorted(list(set(cellpair)))
        
#         all_pathway = self.enrich_result['gmt']
#         all_pathway = {k.replace(re.search('hsa[0-9]* |mmu[0-9]* ', k).group(), '') : all_pathway[k] for k in all_pathway.keys()}
#         all_pathway_names = list(sorted(list(all_pathway.keys())))

#         # creating menu with them 
#         pathway_vars = widgets.Select(
#                         description = 'Pathway:',
#                         options=all_pathway_names,
#                         layout={'width': '90%'}, # If the items' names are long
#                         disabled=False,
#                     )
        
#         sender_receiver_sel = widgets.SelectMultiple(options=sender_receiver,
#                               layout=widgets.Layout(width='90%'))
#         cellpair_sel = widgets.SelectMultiple(options=cellpair,
#                               layout=widgets.Layout(width='90%'))

#         # button, output, function and linkage
#         senser_receiver_butt = widgets.Button(description='CLICK to Print Pathway for Sensor~Receiver Cell',
#                               layout=widgets.Layout(width='90%'))
#         cellpair_butt = widgets.Button(description='CLICK to Print Pathway for Sender Cell -> Receiver Cell',
#                               layout=widgets.Layout(width='90%'))
#         senser_receiver_pathway_butt = widgets.Button(description='CLICK to show enrichment curve for Sensor~Receiver Cell',
#                               layout=widgets.Layout(width='90%'))
#         cellpair_pathway_butt = widgets.Button(description='CLICK to show enrichment curve for Sender Cell -> Receiver Cell',
#                               layout=widgets.Layout(width='90%'))
        
#         outt = widgets.Output()

#         def _one_single_clicked(b, Type):
#             with outt:
#                 clear_output()
#                 print('WARINING: Running, Please do not reflesh until you see the figure!')
#                 pairs = sender_receiver_sel.value if Type == 'sensor' else cellpair_sel.value
#                 if len(pairs) == 1:
#                     self.pathway_scatter( 
#                             a_pair = pairs[0], 
#                             pval_cutoff=0.05, 
#                             ES_cutoff=0,
#                             cmap = 'cool',
#                             vmax = None,
#                             vmin = None,
#                             figsize = 'auto',
#                             title = '',
#                             maxSize = 500,
#                             minSize = 15,
#                             save = None,
#                             show_plot = True)

#                 elif len(pairs) == 2:
#                     self.pathway_stacked_bar(
#                                 pair1 = pairs[0],
#                                 pair2 = pairs[1],
#                                 pval_cutoff=0.05, 
#                                 ES_cutoff=0,
#                                 cmap = 'spring_r',
#                                 vmax = None,
#                                 vmin = None,
#                                 figsize = 'auto',
#                                 title = '',
#                                 maxSize = 500,
#                                 minSize = 15,
#                                 colors = ['#CC6677', '#1E90FF'],
#                                 save = None,
#                                 show_plot = True)
#                 else:
#                     self.pathway_multi_dot(
#                                 pairs = pairs, 
#                                 pval_cutoff=0.05, 
#                                 ES_cutoff=0,
#                                 cmap = 'Spectral_r',
#                                 vmax = None,
#                                 vmin = None,
#                                 node_size_norm = (20, 100),
#                                 figsize = 'auto',
#                                 title = '',
#                                 maxSize = 500,
#                                 minSize = 15,
#                                 save = None,
#                                 show_plot = True)


#         def _enrich_clicked(b, Type):
#             with outt:
#                 clear_output()
#                 print('WARINING: Running, Please do not reflesh until you see the figure!')

#                 pairs = sender_receiver_sel.value if Type == 'sensor' else cellpair_sel.value
#                 a_pair = pairs[0] # only one used
#                 print(a_pair)
#                 ## pathway value
#                 pathway = pathway_vars.value
#                 print(pathway)
#                 ## extract ES and padj
#                 ##
#                 cellpairs = list(self.enrich_result['cellpair_res'].keys())
#                 sensor_receivers = list(self.enrich_result['sensor_res'].keys())

#                 ## plot
#                 self.pathway_ES_plot(a_pair = a_pair, 
#                               pathway = pathway,
#                               figsize = (8, 3.5),
#                               dot_color = '#1874CD',
#                               curve_color = 'black',
#                               title='',
#                               save = None,
#                               show_plot = True,
#                               return_fig = False,
#                               return_data = False
#                              )
               
        
#         senser_receiver_butt.on_click(functools.partial(_one_single_clicked, Type = 'sensor'))
#         # display for sensor in receiver
#         box1 = widgets.VBox([sender_receiver_sel, senser_receiver_butt],
#                             layout=widgets.Layout(width='50%'))

#         cellpair_butt.on_click(functools.partial(_one_single_clicked, Type = 'cellpair'))
#         # display for sender to receiver
#         box2 = widgets.VBox([cellpair_sel, cellpair_butt],
#                             layout=widgets.Layout(width='50%'))
        
#         box3 = widgets.VBox([pathway_vars,
#                              widgets.HBox([senser_receiver_pathway_butt,
#                                            cellpair_pathway_butt])])
#         senser_receiver_pathway_butt.on_click(functools.partial(_enrich_clicked, Type = 'sensor'))
#         cellpair_pathway_butt.on_click(functools.partial(_enrich_clicked, Type = 'cellpair'))

#         mk = Markdown("""<b>Select one or multiple to visulize</b>""")
#         display(mk, widgets.VBox([widgets.HBox([box1, box2]), box3, outt]))

            
            
            
            
            
            
            
            
            
            
