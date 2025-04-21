"""
Created by Jude Wells 2025-04-21

HiQBind was downloaded from:
https://figshare.com/articles/dataset/BioLiP2-Opt_Dataset/27430305?file=52379423
on 2025-04-21

First, create a chronological split of the data 2020, 2021 and 2022 are used for testing
all previous results are used for training and validation.

Next, within both the training set and the validation set we do clustering on both the
ligand and the protein.

We will use MMSEQS easy cluster with 30% sequence identity 50% coverage to generate the protein similarity clusters.

For the ligand clustering use a good open source method.

For the training dataset in a similar fashion to the plinder dataset we will iterate over clusters
where the cluster is defined based on the protein sequence only.

For the test dataset we will do a non-redundancy reduction which considers both the protein and the ligand.
For this we sample 1 test example from for each protein-ligand cluster pair.

The validation set will consis of 10% of the training dataset clusters (based on protein sequence only).

"""