import re
import os
import sys
import time
import argparse
import multiprocessing
from datetime import datetime
import pandas as pd
import numpy as np
from cobra.io import read_sbml_model


def info(string):
    """Print timestamped information."""
    current_time = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    print(f"[{current_time}]: {string}")


def cal_bounds_from_exp(dict_gene_exp, reaction):
    """Calculate reaction bounds based on GPR and gene expression."""
    reac_gpr = str(reaction.gene_reaction_rule)
    reac_gpr_expression = solve_reac_gene_exp(reac_gpr, dict_gene_exp)
    
    # Replace gene logic with evaluated expressions
    for gene_pair_str in re.findall(r'\(.*?\)', reac_gpr_expression):
        gene_pair_str = gene_pair_str.strip("()")
        if len(gene_pair_str.split(' ')) < 3:
            pair_exp = return_min_or_max(gene_pair_str)
        else:
            pair_exp = solve_single_string(gene_pair_str)
        reac_gpr_expression = reac_gpr_expression.replace(f"({gene_pair_str})", str(pair_exp), 1)
    
    return float(solve_single_string(reac_gpr_expression))


def return_min_or_max(gene_string_pair):
    """Evaluate expression pairs by 'OR' and 'AND' logic."""
    gene_string_pair = gene_string_pair.strip("()")
    values = list(map(float, gene_string_pair.replace(' and ', ' or ').split(' or ')))
    return max(values) if ' or ' in gene_string_pair else min(values)


def get_gene_name(Id):
    """Retrieve gene symbol from model given gene ID."""
    try:
        ann = model.genes.get_by_id(Id).annotation
        symbol_index = next(k for k in ann if k.endswith('symbol'))
        return ann.get(symbol_index, '')
    except KeyError:
        return ''


def solve_reac_gene_exp(reac_gpr, dict_gene_exp):
    """Replace gene IDs in GPR with expression values from dict_gene_exp."""
    reac_gpr_exp = [
        f"({dict_gene_exp.get(get_gene_name(gene.strip('()')), 0)})" if gene not in ['and', 'or'] else gene
        for gene in reac_gpr.split(' ')
    ]
    return ' '.join(reac_gpr_exp)


def solve_single_string(reac_gpr_expression):
    """Evaluate a GPR expression string iteratively until a single value remains."""
    while 'and' in reac_gpr_expression or 'or' in reac_gpr_expression:
        match = re.search(r'(\d+(\.\d+)? (and|or) \d+(\.\d+)?)', reac_gpr_expression)
        if match:
            gene_pair_str = match.group(0)
            pair_exp = return_min_or_max(gene_pair_str)
            reac_gpr_expression = reac_gpr_expression.replace(gene_pair_str, str(pair_exp), 1)
        else:
            break
    return reac_gpr_expression


def _run_single_fba(sample_id):
    """Run FBA for a single sample."""
    info(f"FBA for sample: {sample_id}")
    s_gene_dict = mat[sample_id].to_dict()
    sample_gem = model.copy()
    med_reacs = set(sample_gem.medium)

    # Define bounds based on expression
    ge_bounds = {
        reac.id: cal_bounds_from_exp(s_gene_dict, reac) for reac in sample_gem.reactions if reac.id not in med_reacs
    }
    for reac_id, bound in ge_bounds.items():
        if bound != 0:
            sample_gem.reactions.get_by_id(reac_id).bounds = (-bound, bound)

    # Run optimization
    fra = sample_gem.optimize()
    react_flux = {r.id: r.flux for r in sample_gem.reactions}
    pd.DataFrame.from_dict(react_flux, orient='index').to_csv(f'{tpath}/{sample_id}.csv')
    del sample_gem


def _fba_(exp_mat, gem_model, thread, tmp='tmp'):
    """Run FBA using expression data and model."""
    samples = exp_mat.columns.tolist()
    global mat, model, tpath
    mat, model, tpath = exp_mat, gem_model, tmp

    # Check or create output directory
    os.makedirs(tmp, exist_ok=True)
    existing_files = {f.replace('.csv', '') for f in os.listdir(tmp)}
    samples = [s for s in samples if s not in existing_files]

    info(f"Processing {len(samples)} samples")
    if thread:
        with multiprocessing.Pool(thread) as pool:
            pool.map(_run_single_fba, samples)
    else:
        for s in samples:
            _run_single_fba(s)

def _run_fba(gem_file, exp_dat, thread = 8, tmp_dir = '_tmp_cobra_fba_'):
    gem = read_sbml_model(gem_file)
    _fba_(exp_mat=exp_dat, gem_model=gem, thread=thread, tmp=tmp_dir)
    # Consolidate results
    flux_res = pd.concat([
        pd.read_csv(f"{tmp_dir}/{s}.csv", index_col=0).rename(columns={'0': s.replace('.csv', '')})
        for s in os.listdir(tmp_dir)
    ], axis=1)
    info("FBA completed")
    return(flux_res)

def get_parameter():
    parser = argparse.ArgumentParser(description="Run FBA with COBRA")
    parser.add_argument('-g', dest='gem_file', type=str, required=True, help='Path to GEM xml file')
    parser.add_argument('-e', dest='exp_file', type=str, required=True, help='Path to gene expression file')
    parser.add_argument('-n', dest='normalize', type=str, default='false', help='Normalize gene expression (true/false)')
    parser.add_argument('-c', dest='thread', type=int, default=1, help='Number of threads to use')
    parser.add_argument('-td', dest='tmp_dir', type=str, default='tmp', help='Temporary directory for output')
    parser.add_argument('-p', dest='prefix', type=str, help='Prefix for output files')
    return parser.parse_args()


def main():
    args = get_parameter()
    gem = read_sbml_model(args.gem_file)
    gem.solver = 'glpk'
    
    # Load and normalize expression data if needed
    exp_dat = pd.read_csv(args.exp_file, index_col=0, compression='gzip' if args.exp_file.endswith('.gz') else None)
    if args.normalize.lower() == 'true':
        exp_dat = np.log10((exp_dat.div(exp_dat.sum(axis=0) / 1e4)) + 1)

    _fba_(exp_mat=exp_dat, gem_model=gem, thread=args.thread, tmp=args.tmp_dir)

    # Consolidate results
    flux_res = pd.concat([
        pd.read_csv(f"{args.tmp_dir}/{s}.csv", index_col=0).rename(columns={'0': s.replace('.csv', '')})
        for s in os.listdir(args.tmp_dir)
    ], axis=1)
    flux_res.to_csv(f"{args.prefix}.fluxres.csv.gz", compression='gzip')
    info("FBA completed")


if __name__ == '__main__':
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupted! Exiting...\n")
        sys.exit(0)
    print(f"Completed in {time.time() - start_time:.2f} seconds")
