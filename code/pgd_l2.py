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
# import torchattacks
from attacks import pgdl2 as attack
from tqdm import tqdm
parser = argparse.ArgumentParser(description='Test PGD_l2 attack')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()

class SmoothedClassifier(nn.Module):
    
    def __init__(self, base_classifier, sigma):
        super(SmoothedClassifier, self).__init__()
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
    smoothe_classifier = SmoothedClassifier(base_classifier, sigma=0.25)
    smoothe_classifier.eval()
    
    # atk = torchattacks.APGDT(base_classifier, norm='L2', eps=0.5)
    # atk = torchattacks.PGDL2(base_classifier, eps=0.5, alpha=0.05, steps=20)
    atk = attack.EOTPGDL2(smoothe_classifier, eps=0.5, alpha=0.1, steps=10, eot_iter=15)
    atk.set_mode_targeted_by_function(target_map_function=lambda images, labels:labels)
    
    dataset = get_dataset(args.dataset, args.split)
    n_cor = 0
    n_adv_cor = 0
    for i in tqdm(range(len(dataset))):

        if i == 100:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        label = torch.tensor(label, dtype=torch.int64).cuda()
        batch = x.repeat((1, 1, 1, 1))
        label = label.repeat((1))

        predictions = smoothe_classifier(batch).argmax(1)
        print(f'[{i}] clean: ', predictions, label, predictions[0]==label[0])
        
        adv_images = atk(batch, label)
        adv_predictions = smoothe_classifier(adv_images).argmax(1)
        print(f'[{i}] adv: ', adv_predictions, label, adv_predictions[0]==label[0])
        print()
        if predictions[0]==label[0]:
            n_cor += 1
        
        if adv_predictions[0]==label[0]:
            n_adv_cor += 1
            
    print('acc: ', n_cor/i)
    print('adv acc: ', n_adv_cor/i)
    