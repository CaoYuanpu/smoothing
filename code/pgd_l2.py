# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import datetime
from architectures import get_architecture

# parser = argparse.ArgumentParser(description='Certify many examples')
parser = argparse.ArgumentParser(description='Test PGD_l2 attack')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, help="noise hyperparameter")
# parser.add_argument("outfile", type=str, help="output file")
# parser.add_argument("--batch", type=int, default=1000, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
# parser.add_argument("--N0", type=int, default=100)
# parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
# parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()


if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):
        (x, label) = dataset[i]
        x = x.cuda()
        print(x.shape)
        input()
    
    