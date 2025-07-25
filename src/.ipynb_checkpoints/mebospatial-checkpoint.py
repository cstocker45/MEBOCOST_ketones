#!/usr/bin/env python

# ================================
# @auther: Rongbin Zheng
# @email: Rongbin.Zheng@childrens.harvard.edu
# @date: Nov 2024
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
import mebocost_test2.mebocost as mb

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))

def load_obj(path):
    """
    read mebocost object
    """
    file = open(path,'rb')
    dataPickle = file.read()
    file.close()
    obj_vars = cPickle.loads(dataPickle)
    mebocost_obj = create_obj(exp_mat = obj_vars['exp_mat'],
                        adata = obj_vars['adata'],
                        cell_ann = obj_vars['cell_ann'],
                        group_col = obj_vars['group_col'],
                        condition_col = obj_vars['condition_col'],
                        config_path = obj_vars['config_path'],
                       )
    mebocost_obj.__dict__ = obj_vars

    return mebocost_obj

def save_obj(obj, path = 'mebospatial_result.pk', filetype = 'pickle'):
    """
    save object to pickle
    """
    file = open(path,'wb')
    file.write(cPickle.dumps(obj.__dict__))
    file.close()



class create_obj:
    """
    Spatial function in MEBOCOST to predict metabolite-based cell-cell communication (mCCC) using spatial transcriptomics.

    Params
    -------
    dtype: 
        spatial_sc or spatial_spot, spatial_sc for single-cell level spatial data, spatial_spot for spot-level spatial data
    
    model_method: 
        correlation, sc_copresence, cellgroup_copresence. correlation was recommended for spot-level spatial data, such as 10x Visium. sc_copresence and cellgroup_copresence were both recomended for single cell level spatial data. cellgroup_copresence for spatial data with high drop-out rates, such as Visium-HD.
        
    xy_cord_col:
        a type, specify column names of x and y cordinates in cell_ann or adata.obs, such as ('X', 'Y'). Ignore this argument if spatial info is in adata.obsm.
    """
    def __init__(self,  
                adata=None, 
                exp_mat=None, 
                cell_ann=None,
                xy_cord_col=None,
                group_col=None,
                condition_col=None,
                library_key = 'spatial',
                dtype = 'spatial_spot',
                model_method = 'correlation',
                dist_threshold = 'auto',
                species = 'human',
                met_mat=pd.DataFrame(), 
                config_path=None,
                cutoff_exp='auto',
                cutoff_met='auto',
                cutoff_prop=0.15,
                sensor_type=['Receptor', 'Transporter', 'Nuclear Receptor'],
                thread = 1
                ):
        tic = time.time()
        self.adata = adata
        self.exp_mat = exp_mat
        ## add a column "cell_group" if successfull
        if (self.exp_mat is None and cell_ann is None) and (self.adata is not None):
            cell_ann = adata.obs.copy()
        self.cell_ann = cell_ann
        if group_col:
            if group_col not in cell_ann.columns.tolist():
                raise KeyError('group_col: %s is not in cell_ann columns, it should be one of %s'%(group_col, cell_ann.columns.tolist()))
            else:
                self.group_col = group_col
                # cell_ann['cell_group'] = cell_ann[group_col].tolist()
        else:
            self.group_col = group_col
            
        if not condition_col or condition_col in cell_ann.columns.tolist():
            self.condition_col = condition_col
        else:
            raise KeyError('condition_col: %s is not in cell_ann columns, it should be one of %s'%(condition_col, cell_ann.columns.tolist()))

        if self.condition_col and self.group_col:
            group_names = self.cell_ann[self.condition_col].astype('str').str.replace('~', '_')+' ~ '+self.cell_ann[self.group_col].astype('str').str.replace('~', '_')
            self.cell_ann['cell_group'] = group_names
        else:
            group_names = []
        
        self.dist_threshold = dist_threshold
        self.met_mat = met_mat
        self.species = species
        self.config_path = config_path
        self.cutoff_exp = cutoff_exp
        self.cutoff_met = cutoff_met
        self.cutoff_prop = cutoff_prop
        self.sensor_type = sensor_type
        self.thread = thread
        
        ## initial
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
            try:
                self.spatial_cord = self.adata.obsm[library_key][:,:2]
            except:
                self.spatial_cord = self.cell_ann[list(xy_cord_col)].to_numpy()
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

        self.group_names = list(np.unique(group_names))
        
    
    def _load_config_(self):
        """
        load config and read data from the given path based on given species
        """
        ## the path of config file
        info('Load config and read data based on given species [%s].'%(self.species))
        if self.config_path:
            if not os.path.exists(self.config_path):
                raise KeyError('ERROR: the config path is not exist!')
            config = mb._read_config(conf_path = self.config_path)
            ## common
            self.met_ann = pd.read_csv(config['common']['hmdb_info_path'], sep = '\t')
            ## depends on species
            if self.species == 'human':
                self.met_enzyme = pd.read_csv(config['human']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['human']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
                self.gene_network = pd.read_csv(config['human']['gene_network_path'], index_col = 0, compression = 'gzip')
                self.gmt_path = config['human']['kegg_gmt_path']
                self.gem_path = config['human']['gem_path']
            elif self.species == 'mouse':
                self.met_enzyme = pd.read_csv(config['mouse']['met_enzyme_path'], sep = '\t')
                met_sensor = pd.read_csv(config['mouse']['met_sensor_path'], sep = '\t')
#                 met_sensor['gene'] = met_sensor['Gene_name'].apply(lambda x: x.split('[')[0])
                self.met_sensor = met_sensor
                self.gene_network = pd.read_csv(config['mouse']['gene_network_path'], index_col = 0, compression = 'gzip')
                self.gmt_path = config['mouse']['kegg_gmt_path']
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











