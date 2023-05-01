import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import at_train
from src.at_dataset import Multimodal_Datasets, compute_metrics, get_num_classes


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT',
                    help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--dataset', type=str, default='IEMOCAP',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='/mnt/bellago/mt_data',
                    help='path for storing the dataset')

# Roberta
parser.add_argument('--speaker_mode', type=str, default='upper',
                    help='roberta speaker name in capital')
parser.add_argument('--num_past_utterances', type=int, default=00,
                    help='roberta speaker name in capital')
parser.add_argument('--num_future_utterances', type=int, default=0,
                    help='roberta speaker name in capital')
parser.add_argument('--model_checkpoint', type=str, default='none',
                    help='roberta speaker name in capital')
parser.add_argument('--root_dir', type=str, default="/mnt/hard2/bella",
                    help='roberta speaker name in capital')

            
# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.25,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.45,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.45,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.3,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.2,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.2,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=4,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=2,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')


# Logistics
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
#parser.add_argument('--no_cuda', action='store_true',
                    #help='do not use cuda')
parser.add_argument('--name', type=str, default='mulerc_1',
                    help='name of the trial (default: "mult")')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset=args.dataset
#dataset = str.lower(args.dataset.strip())
#valid_partial_mode = args.lonly + args.vonly + args.aonly
valid_partial_mode = args.vonly + args.aonly

'''
if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")
'''
if valid_partial_mode == 0:
    args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'IEMOCAP': 4
}

criterion_dict = {
    'IEMOCAP': 'CrossEntropyLoss'
}

torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available():
    print('cuda')
    torch.cuda.manual_seed(args.seed)
    use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")
train_path = os.path.join(args.root_dir, args.dataset) + 'train_atemoberta_0past.dt'
valid_path = os.path.join(args.root_dir, args.dataset) + 'valid_atemoberta_0past.dt'
test_path = os.path.join(args.root_dir, args.dataset) + 'test_atemoberta_0past.dt'
if not os.path.exists(train_path):
    print("  - Creating new train data")
    train_data = Multimodal_Datasets(
        DATASET=args.dataset,
        SPLIT="train",
        speaker_mode=args.speaker_mode,
        num_past_utterances=args.num_past_utterances,
        num_future_utterances=args.num_future_utterances,
        model_checkpoint=args.model_checkpoint,
        ROOT_DIR=args.root_dir,
        SEED=args.seed,
        )
    valid_data = Multimodal_Datasets(
        DATASET=args.dataset,
        SPLIT="val",
        speaker_mode=args.speaker_mode,
        num_past_utterances=args.num_past_utterances,
        num_future_utterances=args.num_future_utterances,
        model_checkpoint=args.model_checkpoint,
        ROOT_DIR=args.root_dir,
        SEED=args.seed,
        )
    test_data = Multimodal_Datasets(
        DATASET=args.dataset,
        SPLIT="test",
        speaker_mode=args.speaker_mode,
        num_past_utterances=args.num_past_utterances,
        num_future_utterances=args.num_future_utterances,
        model_checkpoint=args.model_checkpoint,
        ROOT_DIR=args.root_dir,
        SEED=args.seed,
        )
    torch.save(train_data, train_path)
    torch.save(valid_data, valid_path)
    torch.save(test_data, test_path)
else:
    print("  - Found cached {split} data")
    train_data = torch.load(train_path)
    valid_data = torch.load(valid_path)
    test_data = torch.load(test_path)


#print(len(train_data), len(valid_data), len(test_data)) 
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=train_data.collater)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=valid_data.collater)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=test_data.collater)

print('Finish loading the data....')
#if not args.aligned:
    #print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
#hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
#hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
#hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict[dataset]
hyp_params.criterion = criterion_dict[dataset]


if __name__ == '__main__':
    test_loss = at_train.initiate(hyp_params, train_loader, valid_loader, test_loader)

