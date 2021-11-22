LEDG
=====

This repository is modified from the source code (https://github.com/IBM/EvolveGCN) of paper Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), in AAAI, 2020.

We thank the authors of EvolveGCN for well-written codes.

## Data

URLs to download the data we used in the paper:

- stochastic block model: This data comes from https://github.com/IBM/EvolveGCN/tree/master/data
- bitcoin OTC: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
- bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
- uc_irvine: Downloadable from http://konect.uni-koblenz.de/networks/opsahl-ucsocial
- autonomous systems: Downloadable from http://snap.stanford.edu/data/as-733.html
- reddit hyperlink network: Downloadable from http://snap.stanford.edu/data/soc-RedditHyperlinks.html
- brain: Downloadable from http://tinyurl.com/y6d74mmv

For downloaded datasets please place them in the 'data' folder.

## Requirements
  * PyTorch 1.0 or higher
  * Python 3.6
  * PyTorch_Geometric

## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file ./experiments/parameters_auto_syst_meta_gcn.yaml
```

will run the experiments of using GCN w/ LEDG on the autonomous system dataset.

The yaml files in the 'experiment' folder contain the hyperparameters for reproducing the results in our paper. 

