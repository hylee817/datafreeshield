import os
import time
import subprocess
import numpy as np
import sys
import glob

def run_or_wait(command):
    # print(command)
    not_done = True
    while not_done:
        try:
            output = subprocess.check_output(command,encoding='utf-8')#command)
            print(command,"is good")
            not_done = False
            time.sleep(2)
        except:
            print(command,"is wating to be assigned to queue...")
            time.sleep(60)


if __name__ == '__main__':

    ckpt = sys.argv[1]
    exp_name = sys.argv[2]
    distance = sys.argv[3]
                        
    csv_file = "result.csv"

    for pth in glob.glob(ckpt+"*/"):
        # model weight should be named "robust_model.pt"
        if not os.path.exists(pth+"robust_model.pt"):
            print("no weight here! {}".format(pth))
            continue
        
        ## ------------ set epsilon -----------##    
        if distance == "Linf":
            if "_4_kd" in pth: eps=4
            elif "_8_kd" in pth: eps=8
            elif "_2_kd" in pth: eps=2
            elif "_6_kd" in pth: eps=6
            else: 
                continue
                # raise Exception(pth)
        elif distance == "L2":
            if "_128_kd" in pth: eps=128
            else:
                raise Exception(pth)
        else:
            raise Exception("Please specify either Linf or L2")

        ## ------------ name model -----------##
        if "resnet20" in pth: model="resnet20"
        elif "resnet56" in pth: model="resnet56"
        elif "wrn28_10" in pth: model="wrn28_10"
        # medmnist
        elif "resnet18" in pth: model="resnet18"
        elif "resnet50" in pth: model="resnet50"
        else: raise Exception(pth)

        ## ------------ name dataset -----------##
        if "cifar100" in pth: dataset ="cifar100"
        elif "cifar10" in pth: dataset="cifar10"
        elif "svhn" in pth: dataset="svhn"
        #medmnist
        elif "tissue" in pth: dataset="tissuemnist"
        elif "blood" in pth: dataset="bloodmnist"
        elif "derma" in pth: dataset="dermamnist"
        elif "path" in pth: dataset="pathmnist"
        elif "oct" in pth: dataset="octmnist"
        elif "organa" in pth: dataset="organamnist"
        elif "organc" in pth: dataset="organcmnist"
        elif "organs" in pth: dataset="organsmnist"
        else: raise Exception(pth)

        ## ------------ submit script to slurm -----------##
        run_or_wait(["sbatch", "eval.sh",
            str(dataset), \
            str(model), \
            str(eps), \
            str(pth), \
            str(exp_name), \
            csv_file,\
            str(distance)])
  
