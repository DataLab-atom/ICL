from train import train_baseline_syn
from train_causal import train_causal_syn
from opts import setup_seed
import torch
import opts
import os
import utils
import pdb
import time
import warnings
warnings.filterwarnings('ignore')

def main():
    args = opts.parse_args()
    args.hidden = 32
    save_path = args.data_root
    os.makedirs(save_path, exist_ok=True)
    try:
        dataset = torch.load(save_path + "/syn_dataset.pt")
    except:
        dataset = utils.graph_dataset_generate(args, save_path)
    train_set, val_set, test_set, the = utils.dataset_bias_split(dataset, args, bias=args.bias, split=[7, 1, 2], total=args.data_num * 4)
    group_counts = utils.print_dataset_info(train_set, val_set, test_set, the)
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        train_causal_syn(train_set, val_set, test_set, model_func=model_func, args=args)
    else:
        assert False

if __name__ == '__main__':
    main()
