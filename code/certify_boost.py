# evaluate a smoothed classifier on a dataset
import argparse
import os
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
import torch.nn as nn
import datetime
from architectures import get_architecture

from attacks import pgdl2 as attack



parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

certify_res_file_path = "/home/ymc5533/smoothing/result/certify_cifar10_0.25"

class BoostClassifier(nn.Module):
    
    def __init__(self, base_classifier, sigma):
        super(BoostClassifier, self).__init__()
        self.base_classifier = base_classifier
        self.sigma = sigma

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn_like(x, device=x.device) * self.sigma
        return base_classifier(x+noise)
    
if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # creat boost classifier
    boost_classifier = BoostClassifier(base_classifier, sigma=args.sigma)
    boost_classifier.eval()
    
    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # read certified result
    certify_res_file = open(certify_res_file_path)
    certify_res_file.readline()

    
    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime\tperturbation", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        
        
        res = certify_res_file.readline()
        res = res.split('\t')
        label = int(res[1])
        predict = int(res[2])
        radius = float(res[3])
        
        if label == predict:
            before_time = time()
            
            # pre_p_a_lower = norm.cdf(radius / args.sigma)

            x_ = x.repeat((1, 1, 1, 1))
            target = torch.tensor(label, dtype=torch.int64)
            target = target.repeat((1))
            atk = attack.EOTPGDL2(boost_classifier, eps=radius, alpha=2*radius/10, steps=10, eot_iter=10)
            atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:labels)
            x_adv = atk(x_.cuda(), target.cuda())

            print(torch.linalg.norm((x_adv - x_.cuda()).detach()[0]))
            input()
            
            # certify the prediction of g around x_adv
            prediction_adv, radius_adv = smoothed_classifier.certify(x_adv, args.N0, args.N, args.alpha, args.batch)
            after_time = time()
            correct = int(prediction_adv == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction_adv, radius_adv, correct, time_elapsed), file=f, flush=True)
        else:
            print('\t'.join(res), file=f, flush=True)

    f.close()
