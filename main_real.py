from datasets import get_dataset
from train_causal import train_causal_real
import opts
import pdb
import warnings
warnings.filterwarnings('ignore')
import time
from train import train_baseline_real 
def main():
    args = opts.parse_args()
    dataset_name, feat_str, _ = opts.create_n_filter_triples([args.dataset])[0]
    dataset = get_dataset(dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
    print(dataset)
    model_func = opts.get_model(args)
    if args.model in ["GIN","GCN", "GAT"]:
        train_baseline_real(dataset, model_func, args)
    elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
        train_causal_real(dataset, model_func, args)
    else:
        assert False
    
if __name__ == '__main__':
    main()