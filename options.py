# _*_ coding: utf-8 _*_
# This file is created by C. Zhang for personal use.
# @Time         : 18/08/2022 16:03
# @Author       : tl22089
# @File         : options.py
# @Affiliation  : University of Bristol
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=20,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate')
    parser.add_argument('--process', type=str, default='stft',
                        help='stft or RAW_IQ ')
    parser.add_argument('--datasettype', type=str, default='mat',
                        help='dat or mat')
    parser.add_argument('--data_choice', type=str, default='Trainday1_Testday1',
                        help='Different dataset splits are used for training and testing others:Trainday1_Testday1 ,Trainday1_Testday2 or Trainadapt_Testday2')
    parser.add_argument('--Base_or_Tri', type=str, default='TripletCNN',
                        help='Whether to use a ternary loss function (default: BaselineCNN or TripletCNN)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='ResNet18', help='model name')
    parser.add_argument('--num_channels', type=int, default=True, help="number \
                        of channels of imgs")
    parser.add_argument('--ws', type=int, default=8192, help="window size")
    parser.add_argument('--margin', type=int, default=0.05, help="window size")
    parser.add_argument('--snr_low', type=int, default=20, help="low snr")
    parser.add_argument('--snr_high', type=int, default=80, help="high snr")
    parser.add_argument('--out_dim', type=int, default=128, help="output size for metric learning")
    # other arguments
    parser.add_argument('--signal', type=str, default='WIFI', help="name \
                        of dataset")
    parser.add_argument('--type', type=str, default='non-metric', help="name \
                            of dataset")
    parser.add_argument('--bs_classes', type=int, default=18, help="number \
                        of classes of BSs")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--transfer', default=1, help="Transfer learning or not")
    parser.add_argument('--noise', default=1, help="add or not add noise to the training samples")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    return args