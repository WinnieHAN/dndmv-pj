import argparse

def parse_args(type=0):
    if type == 0:
        parser = argparse.ArgumentParser(description='PyTorch lndmv')
        parser.add_argument('--port', default=23330, type=int, help='port')
        parser.add_argument('--acc_idx', default=1, type=int, help='acc_idx')
        parser.add_argument('--pre_acc_idx', default=0, type=int, help='acc_idx')
        # parser.add_argument('----top', default='0', type=int, help='pre_acc_idx')
        parser.add_argument('--epochs', default=15, type=int,help='epochs')
        parser.add_argument('--lr', default=0.03, type=float, metavar='LR', help='lr')
        parser.add_argument('--dim1', default=23, type=int, help='dim1')
        parser.add_argument('--dim2', default=10, type=int, help='dim2')
        parser.add_argument('--sleep', default=0, type=int, help='sleep')
        parser.add_argument('--is_manually_tagging', default=0, type=int, help='is_manually_tagging')
        parser.add_argument('--n_clusters', default=35, type=int, help='n_clusters')
        parser.add_argument('--clustering_linkage', default=0, type=int, help='clustering_linkage')
        parser.add_argument('--batch_size_nn', default=200, type=int, help='batch_size_nn')
        args = parser.parse_args()
        return args
    elif type == 1:
        parser = argparse.ArgumentParser(description='PyTorch lndmv')
        parser.add_argument('--port', default=23330, type=int, help='port')
        # args = parser.parse_args()
        return parser

parsing = parse_args(type=1)
a = parsing.parse_args("python --port 3")
# args = main1("--port 23330 --acc_idx 0 --pre_acc_idx 100000 --is_manually_tagging 0 --clustering_linkage 0 --n_clusters 0 --dim1 23 --sleep 0")
print('sssss  ', a)