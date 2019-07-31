import argparse

def parse_args(type):
    parser = argparse.ArgumentParser(description='PyTorch lndmv')
    parser.add_argument('--port', default=23330, type=int, help='port')
    parser.add_argument('--acc_idx', default=1, type=int, help='acc_idx')
    parser.add_argument('--pre_acc_idx', default=0, type=int, help='acc_idx')
    parser.add_argument('--epochs', default=20, type=int, help='epochs')  # 20
    parser.add_argument('--sleep', default=10, type=int, help='sleep')
    parser.add_argument('--is_manually_tagging', default=0, type=int, help='is_manually_tagging')
    parser.add_argument('--n_clusters', default=35, type=int, help='n_clusters')
    parser.add_argument('--clustering_linkage', default=0, type=int, help='clustering_linkage')
    parser.add_argument('--batch_size_nn', default=10, type=int, help='batch_size_nn')
    if type == 0:
        parser.add_argument('--lr', default=0.03, type=float, metavar='LR', help='lr')
        parser.add_argument('--dim1', default=23, type=int, help='dim1')
        parser.add_argument('--dim2', default=10, type=int, help='dim2')
    elif type == 1: # pytorch
        parser.add_argument('--chd_dropout_p', default=0.5, type=float, metavar='chd_dropout_p', help='chd_dropout_p')
        parser.add_argument('--chd_lr', default=0.01, type=float, metavar='chd_lr', help='chd_lr')
        parser.add_argument('--chd_head_lstm_dim', default=10, type=int, help='chd_head_lstm_dim')
        parser.add_argument('--chd_direct_dim', default=10, type=int, help='chd_direct_dim')
        parser.add_argument('--chd_lstm_hidden_dim', default=10, type=int, help='chd_lstm_hidden_dim')
        parser.add_argument('--chd_softmax_layer_dim', default=10, type=int, help='chd_softmax_layer_dim')
        parser.add_argument('--pascal_idx', default=0, type=int, help='pascal_idx')

    args = parser.parse_args()
    return args