import os,sys
import time, re
import pickle as pk
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
import importlib

try:
    from cobra.io import read_sbml_model, write_sbml_model
    import mebocost_test2.fba_cobra as fba_cobra
    importlib.reload(fba_cobra)
except:
    print('Warning: cobra loading problem for FBA analysis. Please avoid using cobra related functions')



def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))

def info(string):
    """
    print information
    """
    today = datetime.today().strftime("%B %d, %Y")
    now = datetime.now().strftime("%H:%M:%S")
    current_time = today + ' ' + now
    print("[{}]: {}".format(current_time, string))
    

def cal_bounds_from_exp(dict_gene_exp, reaction):
    '''
    Solves the GPR equation for input reaction by plugging in 
    expression values from input dictionary
    '''
    gene_to_exp = {}
    reac_gpr = str(reaction.gpr)

    reac_gpr_expression = solve_reac_gene_exp(reac_gpr, dict_gene_exp)

    substr_list = re.findall(r'\(.*?\)', reac_gpr_expression)
#     print(substr_list)

    # created a single string with logic
    for substr_list_ele in substr_list:
        gene_pair_str = substr_list_ele
        if substr_list_ele.count('(') > substr_list_ele.count(')'):
            gene_pair_str = substr_list_ele[1:]
        elif substr_list_ele.count('(') < substr_list_ele.count(')'):
            gene_pair_str = substr_list_ele[:-1]

        if len(gene_pair_str.split(' ')) < 3:
            pair_exp = return_min_or_max(gene_pair_str)
        else:
            pair_exp = solve_single_string(gene_pair_str)

        reac_gpr_expression = reac_gpr_expression.replace(gene_pair_str, str(pair_exp), 1)

    reac_gpr_expression = solve_single_string(reac_gpr_expression)

    return float(reac_gpr_expression)


def return_min_or_max(gene_string_pair):
    '''
    Returns maximum between two gene expressions related by 'OR'
    and minimum between two gene expressions related by 'AND'
    '''
    gene_string_pair = gene_string_pair.replace('(', "")
    gene_string_pair = gene_string_pair.replace(')', "")

    if ' or ' in gene_string_pair:
        str_split = gene_string_pair.split(' or ', 1)
        return max(float(str_split[0]), float(str_split[1]))
    else:
        str_split = gene_string_pair.split(' and ', 1)
        return min(float(str_split[0]), float(str_split[1])) 


def get_gene_name(Id):
    try:
        ann = model.genes.get_by_id(Id).annotation
        symbol_index = [x for x in ann.keys() if x.endswith('symbol')][0]
        symbol = ann.get(symbol_index, '')
        return(symbol)
    except:
        return('')

def solve_reac_gene_exp(reac_gpr, dict_gene_exp):
    '''
    Reads GPR information in model and replaces gene names 
    with expression
    '''
    reac_gpr_split = reac_gpr.split(' ')

    reac_gpr_exp = []
    for str_ele in reac_gpr_split:
        if str_ele not in ['and', 'or']:
            if '(' in str_ele:
                reac_gpr_exp += ['(' + str(dict_gene_exp.get(get_gene_name(str_ele[1:]), 0))]
            elif ')' in str_ele:
                reac_gpr_exp += [str(dict_gene_exp.get(get_gene_name(str_ele[:-1]), 0)) + ')']
            else:
                reac_gpr_exp += [str(dict_gene_exp.get(get_gene_name(str_ele), 0))]
        else:
            reac_gpr_exp += [str_ele]

    reac_gpr_expression = ' '. join(reac_gpr_exp)

    return reac_gpr_expression


def solve_single_string(reac_gpr_expression):
    '''
    calculates boundary value from single GPR string
    '''

    while(('and' in reac_gpr_expression) or ('or' in reac_gpr_expression)):
        gene_exp_list = reac_gpr_expression.split(' ')
        gene_pair_str = ' '.join(gene_exp_list[:3])

        pair_exp = return_min_or_max(gene_pair_str)

        reac_gpr_expression = reac_gpr_expression.replace(gene_pair_str, str(pair_exp), 1)

    return reac_gpr_expression


def _test_(i):
    info('FBA for a sample: %s'%i)
    s_gene_dict = mat[i].to_dict()
    sample_gem = model.copy()
    med_reacs = set(sample_gem.medium)
    ## define gene exp bound
    ge_bounds = {ra.id : cal_bounds_from_exp(s_gene_dict, ra) for ra in sample_gem.reactions if ra.id not in med_reacs}
    ge_bounds_nz = {Id:(-ge_bounds[Id], ge_bounds[Id]) for Id in ge_bounds if ge_bounds[Id] != 0}
    for Id in ge_bounds_nz:
        sample_gem.reactions.get_by_id(Id).bounds = ge_bounds_nz[Id]
    ## optimization
#         sample_gem.solver = 'glpk'
    fra = sample_gem.optimize()
    ## extract reaction flux
    react_flux = {r.id : r.flux for r in sample_gem.reactions}
    pd.DataFrame.from_dict(react_flux, orient = 'index').to_csv('%s/%s.csv'%(tpath, i))
    del sample_gem
    # return(react_flux)

def _fba_(exp_mat, gem_model, thread, tmp='_tmp_cobra_fba_'):
    """
    run FBA by taking expression and metabolic map as inputs
    ===================
    gem_model: a Genome Metabolic Model, such as from Human-GEM, Recon, etc
    exp_mat: a pandas data frame for gene expression matrix, where rows are genes, columns are samples
    """
    ## gene expression to dict
    samples = exp_mat.columns.tolist()
    global mat; mat = exp_mat
    global model; model = gem_model
    global tpath; tpath=tmp

    if os.path.exists(tmp):
        exist_samples = [x.replace('.csv', '') for x in os.listdir(tmp)]
        samples = list(set(samples) - set(exist_samples)) if len(exist_samples) > 0 else samples
        info('restarted, %s samples for processing'%len(samples))
    else:
        os.mkdir(tmp)
    if not thread:
        for s in samples:
            _test_(s)
            # fres = _test_(s)
            # pd.DataFrame.from_dict(fres, orient = 'index').to_csv('%s/%s.csv'%(tmp, s))
    else:
        pool = multiprocessing.Pool(thread)
        pool.map(_test_, samples)
        pool.close()
    return

def _cobra_run_(gem_file, avg_exp, thread = 8, tmp_path = '_tmp_cobra_fba_'):
    """
    running cobra for flux analysis
    """
    gem_model = read_sbml_model(gem_file)
    gem_model.solver = 'glpk'

    ## to avoid wierd cell group name, may cause problem when saving result to file
    ## so build alias to cell group name
    samples = avg_exp.columns.copy()
    info(f"Processing {len(samples)} samples")
    dict1 = pd.Series(range(len(samples)), index = samples) ## key = cell group, value = index
    dict2 = pd.Series(samples, index = range(len(samples))) ## value = cell group, key = index
    dict2.index = dict2.index.astype('str')
    avg_exp.columns = [str(dict1.get(x)) for x in avg_exp.columns.tolist()]
    info('Calculate flux rate for each cell group')
    fba_cobra._fba_(exp_mat=avg_exp, gem_model=gem_model, thread=thread, tmp = tmp_path)
    return(dict2)


def _cobra_efflux_influx_(comm_res, gem_model, flux_res, alias):
    """
    focus on transport reaction to calculate efflux and influx
    """
    info('Calculate efflux for sender cells and influx for receiver cells')
    met_reaction_df = []
    for reaction in gem_model.reactions:
        for met in reaction.metabolites:
            met_reaction_df.append([met.id[:-1], met.id, met.name, met.compartment, reaction.id,
                                    reaction.metabolites[met], reaction.compartments, reaction.subsystem])
    met_reaction_df = pd.DataFrame(met_reaction_df,
                                  columns = ['met_id','met_cid', 'met_name',
                                             'met_comp', 'reaction_id', 'direction',
                                             'reaction_comp', 'subsystem'])
    met_reaction_df['reaction_comp'] = ['; '.join(list(x)) for x in met_reaction_df['reaction_comp'].tolist()]
    ## cell surface transport and exchange reaction
    transport_r = met_reaction_df.query('(subsystem == "Transport reactions" and (reaction_comp == "e; c" or reaction_comp == "c; e")) or (subsystem == "Exchange/demand reactions")')
    transport_r_flux_d = flux_res.loc[transport_r['reaction_id'],].apply(lambda col: col * transport_r['direction'].tolist())
    transport_r_flux_d.index = transport_r.index.tolist()
    transport_r = pd.concat([transport_r, transport_r_flux_d], axis = 1)
    exchange_r = transport_r.query('subsystem == "Exchange/demand reactions"')
    transport_r = transport_r.query('subsystem == "Transport reactions"')
    ## calculate efflux and influx
    efflux = {}
    for m in comm_res['Metabolite_Name'].unique().tolist():
        tr = transport_r.loc[(transport_r['met_name'].str.upper() == m.upper()) | 
                transport_r['met_name'].str.upper().isin(alias[m.upper()]),].query('met_comp == "e"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['reaction_id']).drop(tr.columns[:8].tolist(), axis = 1).max()
            efflux[m] = v

    influx = {}
    for m in comm_res['Metabolite_Name'].unique().tolist():
        tr = transport_r.loc[(transport_r['met_name'].str.upper() == m.upper()) | 
                transport_r['met_name'].str.upper().isin(alias[m.upper()]),].query('met_comp == "c"')
        if tr.shape[0] != 0:
            v = tr.drop_duplicates(subset=['reaction_id']).drop(tr.columns[:8].tolist(), axis = 1).max()
            influx[m] = v
    return(efflux, influx)

def _cobra_collect_(comm_res, tmp_path, gem_path, met_ann, sample_id_dict):
    """
    handle cobra to calculate efflux and influx 
    sample_id_dict: key id, values cell group
    """
    gem_model = read_sbml_model(gem_path)
    alias = {str(i).upper():[str(x).upper() for x in str(j).split('; ')] for i,j in met_ann[['metabolite', 'synonyms_name']].values.tolist()}
    ## collect result from cobra folder
    info('Collect flux rate for all cell groups')
    if len(os.listdir(tmp_path)) > 0:
        flux_res = pd.DataFrame()
        for s in os.listdir(tmp_path):
            if s.startswith('.'): # avoid hidden files
                continue
            ss = pd.read_csv(tmp_path+'/'+s, index_col = 0)
            ss.columns = [s.replace('.csv', '')]
            flux_res = pd.concat([flux_res, ss], axis = 1)
        flux_res.columns = [sample_id_dict.get(x) for x in flux_res.columns.tolist()]
    else:
        info('Nothing were found in %s folder, please try run cobra flux again'%(tmp_path))
        sys.exit(1)
    efflux, influx = _cobra_efflux_influx_(comm_res, gem_model, flux_res, alias) ## a dict, key = met, value = a series with cell group as index
    return(efflux, influx)

# def _compass_(tmp_path):
#     """
#     running compass
#     """
#     cmd = 'compass --data $exptsv_path --num-thread $core --species $species --output-dir $output_path --temp-dir $temp_path --calc-metabolites --lambda 0.25'
def _get_one_compass_(compass_folder):
    
    if os.path.exists(compass_folder):
        uptake_path = os.path.join(compass_folder, 'uptake.tsv')
        secret_path = os.path.join(compass_folder, 'secretions.tsv')
        if os.path.exists(uptake_path) and os.path.exists(secret_path):
            uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
            secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
        else:
            uptake_path = os.path.join(compass_folder, 'uptake.tsv.gz')
            secret_path = os.path.join(compass_folder, 'secretions.tsv.gz')
            if os.path.exists(uptake_path) and os.path.exists(secret_path):
                uptake = pd.read_csv(uptake_path, index_col = 0, sep = '\t')
                secretion = pd.read_csv(secret_path, index_col = 0, sep = '\t')
            else:
                raise ValueError('Failed to identify COMPASS output files')
    else:
        raise ValueError('compass_folder path does not exist')
    return(uptake, secretion)

def _get_compass_flux_(compass_folder, compass_met_ann_path, met_ann):  
    
    if isinstance(compass_folder, dict):
        uptake = pd.DataFrame()
        secretion = pd.DataFrame()
        for cond in compass_folder:
            # try:
            uptake_tmp, secretion_tmp = _get_one_compass_(compass_folder[cond])
            # except:
            #     continue
            uptake_tmp.columns = [cond + ' ~ ' + x for x in uptake_tmp.columns.tolist()] 
            secretion_tmp.columns = [cond + ' ~ ' + x for x in secretion_tmp.columns.tolist()] 
            uptake = pd.concat([uptake, uptake_tmp], axis = 1)
            secretion = pd.concat([secretion, secretion_tmp], axis = 1)
    else:
        uptake, secretion = _get_one_compass_(compass_folder)
                    
    ## load compass annotation
    compass_met_ann = pd.read_csv(compass_met_ann_path) #_read_config(self.config_path)['common']['compass_met_ann_path'])
    # compass_rxn_ann = pd.read_csv(_read_config(self.config_path)['common']['compass_rxt_ann_path'])
    ## annotate compass result
    efflux_mat = pd.merge(secretion, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    efflux_mat = pd.merge(efflux_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    efflux_mat = efflux_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    influx_mat = pd.merge(uptake, compass_met_ann[['met', 'hmdbID']],
                            left_index = True, right_on = 'met').dropna()
    influx_mat = pd.merge(influx_mat, met_ann[['Secondary_HMDB_ID', 'metabolite']],
                            left_on = 'hmdbID', right_on = 'Secondary_HMDB_ID')
    influx_mat = influx_mat.drop(['met','hmdbID','Secondary_HMDB_ID'], axis = 1).groupby('metabolite').max()
    return(efflux_mat, influx_mat)

        
