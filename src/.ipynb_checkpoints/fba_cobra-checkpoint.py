import os,sys
import cobra
from cobra.io import read_sbml_model, write_sbml_model
import pandas as pd
import numpy as np
import re
import time, collections
import multiprocessing
from datetime import datetime
import argparse

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
        ## into dict
        # for e,s in zip(*[perm_col, exp_mat.columns.tolist()]):
        #     ## i is index of permute, e is exp shuffling, m is met shuffling
        #     # flux_res[s] = e
        #     pd.DataFrame.from_dict(e, orient = 'index').to_csv('%s/%s.csv'%(tmp, s))
    return

def _run_fba(gem_file, exp_dat, thread = 8, tmp_dir = '_tmp_cobra_fba_'):
    gem = read_sbml_model(gem_file)
    _fba_(exp_mat=exp_dat, gem_model=gem, thread=thread, tmp=tmp_dir)
    # Consolidate results
    flux_res = pd.concat([
        pd.read_csv(f"{tmp_dir}/{s}", index_col=0).rename(columns={'0': s.replace('.csv', '')})
        for s in os.listdir(tmp_dir)
    ], axis=1)
    info("FBA completed")
    return(flux_res)


def get_parameter():
    parser = argparse.ArgumentParser(description="""FBA by Cobra""")

    parser.add_argument('-g', dest='gem_file', type=str, required=True, 
        help='path of GEM xml file')
    parser.add_argument('-e', dest='exp_file', type=str, required=True, 
        help='path of gene expression file (.csv file), rows are genes, columns are samples')
    parser.add_argument('-n', dest='normalize', type=str, required=False, 
        help='true or false, true to normalize count data, false not')
    parser.add_argument('-c', dest='thread', type=str, required=False, 
        help='number of thread used in processing')
    parser.add_argument('-td', dest='tmp_dir', type=str, required=False, 
        help='temp directory for saving reasults for each sample')
    parser.add_argument('-p', dest='prefix', type=str, required=False, 
        help='prefix of output name')

    args = parser.parse_args()
    return(args)

def main():
    args = get_parameter()
    gem_file = args.gem_file
    exp_file = args.exp_file
    normalize = args.normalize
    thread = args.thread
    tmp = args.tmp_dir
    prefix = args.perfix

    gem = read_sbml_model(gem_file)
    gem.solver = 'glpk'
    if exp_file.endwith('.gz'):
        exp_dat = pd.read_csv(exp_file, index_col = 0, compression = 'gzip')
    else:
        exp_dat = pd.read_csv(exp_file, index_col = 0)

    if normalize == 'true':
        colsums = exp_dat.sum() / 1e04
        exp_dat = exp_dat.apply(lambda row: row / colsums, axis = 1)
        exp_dat = np.log10(exp_dat + 1)

    _fba_(exp_mat=exp_dat, gem_model=gem, thread=thread, tmp = tmp)

    if len(os.listdir(tmp)) > 0:
        flux_res = pd.DataFrame()
        for s in os.listdir(tmp):
            ss = pd.read_csv(tmp+'/'+s, index_col = 0)
            ss.columns = [s.replace('.csv', '')]
            flux_res = pd.concat([flux_res, ss], axis = 1)
        flux_res.to_csv(prefix+'.fluxres.csv.gz', compression = 'gzip')
    info('Done in {:.4f} seconds'.format(toc-tic))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrunpt me! ;-) Bye!\n")
        sys.exit(0)

