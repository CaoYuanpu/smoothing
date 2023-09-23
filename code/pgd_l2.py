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
import torchattacks

parser = argparse.ArgumentParser(description='Test PGD_l2 attack')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
args = parser.parse_args()

    
if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    base_classifier.eval()
    
    atk = torchattacks.APGD(base_classifier, norm='L2', eps=0.5)
    
    dataset = get_dataset(args.dataset, args.split)
    n_cor = 0
    n_adv_cor = 0
    for i in range(len(dataset)):

        if i == 100:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        label = torch.tensor(label, dtype=torch.int64).cuda()
        batch = x.repeat((1, 1, 1, 1))
        label = label.repeat((1))

        predictions = base_classifier(batch).argmax(1)
        print('clean: ', predictions, label, predictions[0]==label[0])
        
        adv_images = atk(batch, label)
        adv_predictions = base_classifier(adv_images).argmax(1)
        print('adv: ', adv_predictions, label, adv_predictions[0]==label[0])
        
        if predictions[0]==label[0]:
            n_cor += 1
        
        if adv_predictions[0]==label[0]:
            n_adv_cor += 1
            
    print('acc: ', n_cor/i)
    print('adv acc: ', n_adv_cor/i)
    