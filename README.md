This repo contains pytorch code for the encoder/decoder framework described in "A Deep Learning Framework for Flexible Docking and Design." A description of each folder is given below.

AlphaFold-predicted data for each benchmark used in the paper is available upon request. Result PDB files for each method are also available upon request. See paper for contact information.

Code for inference, training, and data generation is being withheld until formal publication. 

## common

The common folder contains protein specific constants such as residue names, atom types, side-chain dihedral
information, etc. It also contains some helper functions used throughout the code.

## networks

This folder provides implementation of the residue and pair update blocks described in the paper, and an implementation of the encoder/decoder modules (`net.py` is used for both). Encoder and Decoder modules are specified by generating a config from `net_config.py`. An implementation of FAPE loss and rigids from AlphaFold2 is also provided.

## scripts
Contains a folder for each method used in the comparisons during the paper. Code in each folder can be used to generate batched inference scripts for the respective method. Global variables pointing to the respective program will need to be changed. Each is placed at the top of the respective file. All other arguments are passed at the command line.







