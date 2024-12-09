import numpy as np
import pandas as pd
from PIL import Image
import argparse
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from utils.dataset_utils import check, separate_data, split_data, save_file

import os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__)) 

def relabel(c):
    """
    maps hexadecimal class value (string) to a decimal number
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    """

    if c.isdigit() and int(c) < 40:
        return int(c) - 30
    elif int(c, 16) <= 90:  # uppercase
        return int(c, 16) - 55
    else:
        return int(c, 16) - 61  
    
class FemnistDataset(Dataset):
    def __init__(self, data, transform):
        # super(FemnistNiidDataset, self).__init__()

        self.data = data
        self.transforms = transform

        self.transform = transform
        self.images = self.get_images()
        self.targets = self.get_targets()
        
    def __getitem__(self, index):
        return self.images[index], self.targets[index]

    def get_images(self):
        pixel_values = []
        for i in range(len(self.data)):
            path = os.path.join(ROOT_PATH, self.data[i][0])
            image = Image.open(path).convert("L")
            pixel_value = self.transform(image)
            pixel_values.append(pixel_value)
        return pixel_values
    
    def get_targets(self):
        labels = []
        for i in range(len(self.data)):
            label = torch.tensor(relabel(self.data[i][1]))
            labels.append(label)
        return labels
    
    def __len__(self):
        return len(self.data)
    
 
def get_writer_id(data, num_clients):
        images_per_writer = [(row[0],len(row[1])) for row in data]
        images_per_writer.sort(key = lambda x : x[1], reverse=True)
    
        writers = images_per_writer[:num_clients]
        user_ids = [w_id for w_id, count in writers]
    
        return user_ids
    
# Allocate data to users
def generate_femnist(dataset, meta_path, save_path,  num_clients, num_classes, niid, balance, partition):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Setup directory for train/test data
    config_path = os.path.join(save_path, 'config.json')
    train_path = os.path.join(save_path, 'train')
    test_path = os.path.join(save_path, 'test')
    
    if check(config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    data = pd.read_pickle(meta_path)
    writer_ids = get_writer_id(data, num_clients)
    
 
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    for i in range(len(writer_ids)):
        
        user_data = [row[1] for row in data if row[0] == writer_ids[i]][0]
        dataset = FemnistDataset(data = user_data, transform = transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=len(dataset.data), shuffle=False)
  
        for _, train_data in enumerate(dataloader, 0):
            dataset.images, dataset.targets = train_data
   
        X[i] = dataset.images.repeat(1,3,1,1).cpu().detach().numpy()
        y[i] = dataset.targets.cpu().detach().numpy()
        assert len(X[i]) == len(y[i]) , 'the length of images must be equal to the length of targets '
        statistic[i] = (int(i), len(X[i]))
        
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--niid', type=bool, default=True)
    parser.add_argument('--balance', type=bool, default=False)
    parser.add_argument('--partition', type=bool, default=None)
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='FEMNIST')
    parser.add_argument('--meta_file_name', type=str, default='images_by_writer.pkl')
    parser.add_argument('--save_path', type=str, default='FEMNIST')
    args = parser.parse_args()
    
    random.seed(1)
    np.random.seed(1)
    num_classes = 62
   
    meta_path = os.path.join(args.dataset,'intermediate', args.meta_file_name)
    save_path = os.path.join(ROOT_PATH, args.save_path)
    
    generate_femnist(args.dataset, meta_path, save_path, args.num_clients, num_classes, args.niid, args.balance, args.partition)
