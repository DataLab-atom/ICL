# AAAI 2024: A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks
This repository contains code for the AAAI 2024 paper: [A Twist for Graph Classification: Optimizing Causal Information Flow in Graph Neural Networks](#)

## Dependencies

Please setup the environment following Requirements in this [repository](https://github.com/chentingpc/gfn#requirements).
Typically, you might need to run the following commands:

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch-geometric == 2.0.2
pip install torch-scatter  == 2.0.9
pip install torch-sparse == 0.6.15 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install networkx                    
pip install matplotlib  
pip install dgl-cu101
```

## Experiments

### For dir datasets 
```
python main_dir.py --dataset mnist --bias 0.8 --model 'CausalGCN'  --train_model swl
python main_dir.py --dataset mnist --bias 0.85 --model 'CausalGCN'  --train_model mgda 
```
### For TU datasets

```
python main_real.py --model CausalGAT --dataset MUTAG --train_model swl
python main_real.py --model CausalGAT --dataset MUTAG --train_model mgda 
```

### For syn datasets

```
python main_syn.py --model CausalGAT --bias 0.7 --train_model swl
python main_syn.py --model CausalGAT --bias 0.7 --train_model mgda 
```

## Data download
dir datasets can be get in my paper [dir_data_geter](https://github.com/haibin65535/temp/tree/main/dir_data_geter),
TU datasets and syn datasets can be downloaded when you run ``main_real.py`` and ``main_syn.py``.
