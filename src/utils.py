import torch
import os
#from src.dataset import Multimodal_Datasets
from src.dataset import Multimodal_Datasets, compute_metrics, get_num_classes



def get_data(args, dataset, split='train'):
    alignment = 'na'
    data_path = os.path.join(args.data_path, dataset) + '_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print("  - Creating new {split} data")
        data = Dataset(args.data_path, dataset, split)
        torch.save(data, data_path)
    else:
        print("  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    name = name if len(name) > 0 else 'nonaligned_model'

    return name 


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model.state_dict(), '/mnt/hard2/bella/erc/pretrained_models/'+name+'.pt')


def load_model(args, model, name=''):
    name = save_load_name(args, name)
    #print(name)
    model.load_state_dict(torch.load('/mnt/hard2/bella/erc/pretrained_models/'+name+'.pt'))
    return model
