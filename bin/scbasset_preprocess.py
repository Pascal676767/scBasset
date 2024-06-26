#!/usr/bin/env python
import os
import h5py
import anndata
import configargparse
import numpy as np
import pandas as pd
import scanpy as sc
from scbasset.utils import *


def make_parser():
    parser = configargparse.ArgParser(
        description="Preprocess anndata to generate inputs for scBasset.")
    parser.add_argument('--ad_file', type=str,
                        help='Input scATAC anndata. .var must have chr, start, end columns. anndata.X must be in csr format.')
    parser.add_argument('--input_fasta', type=str,
                        help='Genome fasta file.')
    parser.add_argument('--out_path', type=str, default='./processed',
                        help='Output path. Default to ./processed/')
    parser.add_argument('--chromosomes', type=str, nargs='+', default=['chr2', 'chr19'],
                        help='Chromosomes to exclude from the training data set, that will be used for test and validation. Default to chr2 and chr19.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Size of the batch. Default to 1000')

    return parser



def main():
    parser = make_parser()
    args = parser.parse_args()
    
    input_ad = args.ad_file
    input_fasta = args.input_fasta
    output_path = args.out_path
    chromosomes = args.chromosomes
    batch_size = args.batch_size

    
    ad = anndata.read_h5ad(input_ad)

    # sample cells
    data_path = '.'
    os.makedirs(output_path, exist_ok=True)
    seq_len = 1344

    # save anndata
    ad.write('%s/ad.h5ad' % output_path)
    print('Successful writing h5ad file.')

    # save peak bed file
    ad.var.loc[:, ['chr', 'start', 'end']].to_csv('%s/peaks.bed' % output_path, sep='\t', header=False, index=False)
    print('Successful writing bed file.')

    # save train, test, val splits
    train_ids, test_ids, val_ids = split_train_test_val(ad, chromosomes, batch_size)
    with h5py.File('%s/splits.h5' % output_path, "w") as f:
        f.create_dataset("train_ids", data=train_ids)
        f.create_dataset("test_ids", data=test_ids)
        f.create_dataset("val_ids", data=val_ids)
    print('Successful writing split file.')

    # save labels (ad.X)
    m = ad.X.tocoo().transpose().tocsr()
    m_train = m[train_ids, :]
    m_val = m[val_ids, :]
    m_test = m[test_ids, :]
    sparse.save_npz('%s/m_train.npz' % output_path, m_train, compressed=False)
    sparse.save_npz('%s/m_val.npz' % output_path, m_val, compressed=False)
    sparse.save_npz('%s/m_test.npz' % output_path, m_test, compressed=False)
    print('Successful writing sparse m.')

    # save sequence h5 file
    ad_train = ad[:, train_ids]
    ad_test = ad[:, test_ids]
    ad_val = ad[:, val_ids]
    make_h5_sparse(ad, '%s/all_seqs.h5' % output_path, input_fasta, batch_size)
    make_h5_sparse(ad_train, '%s/train_seqs.h5' % output_path, input_fasta, batch_size)
    make_h5_sparse(ad_test, '%s/test_seqs.h5' % output_path, input_fasta, batch_size)
    make_h5_sparse(ad_val, '%s/val_seqs.h5' % output_path, input_fasta, batch_size)


if __name__ == "__main__":
    main()
