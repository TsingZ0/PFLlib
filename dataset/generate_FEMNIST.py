# Generate femnist dataset for Federated Learning settings.
import shutil
from torchvision import datasets, transforms
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, Subset
from argparse import ArgumentParser
import os
import numpy as np

def save_stats_json(dataset_type, stats, outdir, 
                    alpha=None, least_samples=None, 
                    dist=None, blc=None, cc=None, 
                    num_clients=None, seed=None):
    """Save stats to json file. 
    """
    all_stats = {}
    all_stats['dataset_type'] = dataset_type
    if alpha: all_stats['alpha'] = alpha
    if least_samples: all_stats['least_samples'] = least_samples
    if dist: all_stats['distribution'] = dist
    if blc: all_stats['balance'] = blc
    if cc: all_stats['class_per_clients'] = cc
    if num_clients: all_stats['num_clients'] = num_clients
    if seed: all_stats['seed'] = seed
    all_stats['total_train_size'] = sum([stat['train_size'] for k,stat in stats.items()])
    all_stats['total_test_size'] = sum([stat['test_size'] for k,stat in stats.items()])
    all_stats['client_stats'] = stats
    with open(os.path.join(outdir, 'stats.json'), 'w') as f:
        json.dump(all_stats, f, indent=4)


def process_x(raw_data):
    """Transform original 1-D list to 28*28 np array.
    """
    x = np.array(raw_data)
    x = x.reshape(-1, 28, 28)
    x = x.astype(np.float32)
    return x
    
def get_stat(train_dict, test_dict):
    """Get stat for image classification datasets.
    """
    stats = {}
    stats['size'] = len(train_dict['y']) + len(test_dict['y'])
    stats['train_size'] = len(train_dict['y'])
    stats['test_size'] = len(test_dict['y'])
    stats['num_classes'] = len(np.unique(np.concatenate(
        [train_dict['y'], test_dict['y']]
        )))
    stats['num_train_classes'] = len(np.unique(train_dict['y']))
    stats['num_test_classes'] = len(np.unique(test_dict['y']))
    stats['train_class_distribution'] = {i: int(sum(train_dict['y'] == i)) for i in np.unique(train_dict['y']).tolist()}
    stats['test_class_distribution'] = {i: int(sum(test_dict['y'] == i)) for i in np.unique(test_dict['y']).tolist()}
    return stats


if __name__ == '__main__':
    parser = ArgumentParser(description='CIFAR-10 generation for FLamingo')
    # parser.add_argument('--nc', type=int, default=30, help='number of clients')
    parser.add_argument('--seed', type=int, default=2048, help='random seed')
    parser.add_argument('--indir', type=str, default='./utils/leaf_scripts/femnist/data/', help='input dataset directory')
    parser.add_argument('--outdir', type=str, default='../datasets/', help='output dataset directory')
    # Choose how many writers for one client. Default is 2.
    parser.add_argument('-nw', '--num_writers', type=int, default=2, help='Choose how many writers on single client.')
    parser.add_argument('-mts', '--minimum_test_samples', type=int, default=0, help='Minimum number of test samples for a client')
    # usage: bash ./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8
    # non-iid, select 20% percent of clients, keep clients with at least 0 samples, train-test split 80-20
    # Make sure to delete the rem_user_data, sampled_data, test, and train subfolders in the data directory before re-running preprocess.sh
    # usage: python gen_femnist.py --seed 2048 --outdir ../datasets/
    args = parser.parse_args()

    # num_clients = args.nc
    seed = args.seed
    indir = args.indir
    outdir = args.outdir    
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    outdir = os.path.join(outdir, f'femnist')
    train_dir = os.path.join(outdir, 'train')
    test_dir = os.path.join(outdir, 'test')
    if os.path.exists(outdir):
        print("Dataset already generated. Do you want to overwrite it? (y/n)")
        ans = input()
        if ans.lower() != 'y':
            exit(0)
        else:
            shutil.rmtree(outdir)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

        
    # suppose you have already downloaded the dataset 
    # and placed it in the input directory
    raw_train_dir = os.path.join(indir, 'train/')
    raw_test_dir = os.path.join(indir, 'test/')
    train_files = os.listdir(raw_train_dir)
    test_files = os.listdir(raw_test_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    test_files = [f for f in test_files if f.endswith('.json')]
    
    # the files are in json format
    # Travel over the json files and load them into a big dict-like object
    raw_train_data = {}
    raw_test_data = {}
    raw_users = []
    for train_file in train_files:
        tr_f = os.path.join(raw_train_dir, train_file)
        te_f = os.path.join(raw_test_dir, train_file.replace('train', 'test'))
        # print(f"Loading {tr_f} and {te_f}...")
        with open(tr_f) as f:
            temp_f = json.load(f)
            raw_train_data.update(temp_f['user_data'])
            raw_users.extend(temp_f['users'])
        with open(te_f) as f:
            temp_f = json.load(f)
            raw_test_data.update(temp_f['user_data'])
    print("All data loaded...")
    print(f"Total writers: {len(raw_users)}")

    train_data_ = []
    train_data_len = []
    test_data_ = []
    test_data_len = []

    for user in raw_users:
        train_data_.append({'x': process_x(raw_train_data[user]['x']), 'y': np.array(raw_train_data[user]['y'])})
        test_data_.append({'x': process_x(raw_test_data[user]['x']), 'y': np.array(raw_test_data[user]['y'])})
        
    # merge some clients according to args.num_writers
    users_mappings = []
    num_clients = len(train_data_) // args.num_writers
    if args.num_writers > 1:
        sf_idx = np.random.permutation(np.arange(len(train_data_)))
        merged_train_data_ = []
        merged_test_data_ = []
        for i in range(num_clients):
            temp_x, temp_y, temp_u = [], [], []
            for j in range(args.num_writers):
                temp_x.extend(train_data_[sf_idx[i*args.num_writers+j]]['x'])
                temp_y.extend(train_data_[sf_idx[i*args.num_writers+j]]['y'])
                temp_u.append(raw_users[sf_idx[i*args.num_writers+j]])
            merged_train_data_.append({'x':np.array(temp_x), 'y': np.array(temp_y)})
            train_data_len.append(len(temp_y))
            temp_x, temp_y = [], []
            for j in range(args.num_writers):
                temp_x.extend(test_data_[sf_idx[i*args.num_writers+j]]['x'])
                temp_y.extend(test_data_[sf_idx[i*args.num_writers+j]]['y'])
            merged_test_data_.append({'x': np.array(temp_x), 'y': np.array(temp_y)})
            test_data_len.append(len(temp_y))
            users_mappings.append(temp_u)
            print(f"Client {i+1} has {train_data_len[-1], len(temp_y)} samples.")
    else:
        merged_train_data_ = train_data_
        merged_test_data_ = test_data_
        train_data_len = [len(x['y']) for x in train_data_]
        test_data_len = [len(x['y']) for x in test_data_]
        users_mappings = raw_users
        # print
        for i, le in enumerate(train_data_len):
            print(f"Client {i+1} has {train_data_len[i], test_data_len[i]} samples.")
    
    del train_data_, test_data_
    train_data = []
    test_data = []

    inds = sorted(range(len(train_data_len)), key=lambda k: train_data_len[k])
    for ind in inds:
        train_data.append(merged_train_data_[ind])
        test_data.append(merged_test_data_[ind])
    
    stats = {}
    idx = 0
    for i, train_dict in enumerate(train_data):
        if len(test_data[i]['y']) < args.minimum_test_samples:
            continue
        test_dict = test_data[i]
        temp_stat = get_stat(train_dict, test_dict)
        temp_stat['writers'] = users_mappings[i]
        np.savez_compressed(os.path.join(train_dir, f'train{idx}_.npz'), 
                            data={'x':train_dict['x'], 'y':train_dict['y']})
        np.savez_compressed(os.path.join(test_dir, f'test{idx}_.npz'), 
                            data={'x':test_dict['x'], 'y':test_dict['y']})
        stats[idx+1] = temp_stat
        idx += 1
    print(f"Data saved in {outdir}, total {idx} clients generated.")
    save_stats_json('femnist', stats, outdir, seed=seed, dist='niid', num_clients=idx)
    print("Dataset generated successfully!")