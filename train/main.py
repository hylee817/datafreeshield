import argparse
import datetime
import logging
import os
import time
import traceback
import sys
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt

# option file should be modified according to your expriment
from options import Option

from dataloader import DataLoader
from trainer import Trainer

import utils as utils
from pytorchcv.model_provider import get_model as ptcv_get_model

import requests
import csv
from tqdm import tqdm
import random
from slack_token import token

import medmnist
from medmnist import INFO, Evaluator
from models import ResNet18, ResNet50


class Normalize(torch.nn.Module):
	def __init__(self, mean, std):
		super(Normalize, self).__init__()
		self.mean = mean
		self.std = std

	def forward(self, x):
		mu = torch.tensor(self.mean).view(3, 1, 1).cuda()
		std = torch.tensor(self.std).view(3, 1, 1).cuda()
		return (x - mu)/std

class ExperimentDesign:
	def __init__(self, options=None, conf_path=None):
		self.settings = options or Option(conf_path)
		self.train_loader = None
		self.test_loader = None
		self.model = None
		self.model_teacher = None
		self.teacher_wt = self.settings.teacher_wt
		self.advloss_wt = self.settings.advloss_wt
		self.kdloss_wt = self.settings.kdloss_wt
		self.adv_loss = self.settings.advloss
		self.model_name = self.settings.model
		self.from_scratch = self.settings.from_scratch
		
		self.optimizer_state = None
		self.trainer = None
		self.start_epoch = 0
		self.test_input = None
		
		if not self.settings.multigpu: #disable in datacenter
			os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
			os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices
		
		self.logger = self.set_logger()
		# self.settings.paramscheck(self.logger)

		# self.prepare()
		self.set_params()
 
	def set_logger(self):
		logger = logging.getLogger('baseline')
		file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
		console_formatter = logging.Formatter('%(message)s')
		# file log
		file_handler = logging.FileHandler(os.path.join(self.settings.save_path, "train_test.log"))
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(file_formatter)
		
		# console log
		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(logging.INFO)
		console_handler.setFormatter(console_formatter)
		
		logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		
		logger.setLevel(logging.DEBUG)
		return logger

	def set_params(self):
		self._set_gpu()
		self._set_dataloader()
		self._set_model(pretrained=(not self.from_scratch))
		self.print_log()
		self._set_attack()
		self._set_trainer()
	
	def print_log(self):
		# self.logger.info(self.model)
		attrs = vars(self.settings)
		for n, item in attrs.items():
			if n != 'conf': self.logger.info('{}:{}'.format(n,item))

	def _set_gpu(self):
		torch.manual_seed(self.settings.manualSeed)
		torch.cuda.manual_seed(self.settings.manualSeed)
		assert self.settings.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
		cudnn.benchmark = True

	def _set_dataloader(self):
		# create data loader
		data_loader = DataLoader(config=self.settings)
		self.data_config = data_loader.data_config
		self.train_loader, self.test_loader = data_loader.getloader()
		self.test_len = data_loader.get_len()

	def _set_model(self, pretrained=True):
		## general domain
		if self.data_config.alias in ["svhn", "cifar10", "cifar100"]:
			mname = f"{self.model_name}_{self.data_config.alias}"
			print(mname)
			if self.teacher_wt != None:
				self.model = ptcv_get_model(mname, pretrained=False)
				self.model_teacher = ptcv_get_model(mname, pretrained=False)
				model_ckpt = torch.load(self.teacher_wt) #load adv-trained robust weight
				self.model_teacher.load_state_dict(model_ckpt)
				self.model.load_state_dict(model_ckpt)
				print("model weight loaded")
			else:
				try:
					self.model = ptcv_get_model(mname, pretrained=pretrained)
					self.model_teacher = ptcv_get_model(mname, pretrained=True)
				except:
					assert False, "unsupport model: " + self.model_name

		## medmnist
		elif self.data_config.alias == "medmnist":
			n_channels = 3#self.info['n_channels'] #pretrained architecture
			ckpt = torch.load(f"../pretrained/{self.data_config.subclass}/{self.model_name}/model.pt")
			if self.settings.model == "resnet18":
				self.model = ResNet18(in_channels=n_channels, num_classes=self.data_config.n_classes)
				self.model_teacher = ResNet18(in_channels=n_channels, num_classes=self.data_config.n_classes)
			elif self.settings.model == "resnet50":
				self.model = ResNet50(in_channels=n_channels, num_classes=self.data_config.n_classes)
				self.model_teacher = ResNet50(in_channels=n_channels, num_classes=self.data_config.n_classes)
			
			self.model.load_state_dict(ckpt)
			self.model = self.model.eval()
			self.model_teacher.load_state_dict(ckpt)
			self.model_teacher = self.model_teacher.eval()
			print("model loaded")
	

		# append normalize layer for pgd
		self.model = torch.nn.Sequential(Normalize(mean=self.data_config.mean, std=self.data_config.std),self.model)
		self.model_teacher = torch.nn.Sequential(Normalize(mean=self.data_config.mean, std=self.data_config.std),self.model_teacher)

		if self.settings.multigpu:
			self.model = nn.DataParallel(self.model, device_ids=[0,1])
			self.model_teacher = nn.DataParallel(self.model_teacher, device_ids=[0,1])
		self.model.cuda()
		self.model_teacher.cuda()

	
	def _set_attack(self):
		self.step_size = self.settings.step_size
		self.beta = self.settings.beta
		self.gamma = self.settings.gamma
		self.eps = self.settings.eps
		self.steps = self.settings.steps

	def _set_trainer(self):
		# set lr master
		lr_master_S = utils.LRPolicy(self.settings.lr_S,
		                           self.settings.nEpochs,
		                           self.settings.lrPolicy_S)

		params_dict_S = {
			'step': self.settings.step_S,
			'decay_rate': self.settings.decayRate_S
		}
		 
		lr_master_S.set_params(params_dict=params_dict_S)

		# set trainer
		self.trainer = Trainer(
			model=self.model,
			model_teacher=self.model_teacher,
			train_loader=self.train_loader,
			test_loader=self.test_loader,
			test_len=self.test_len,
			data_config=self.data_config,
			lr_master_S=lr_master_S,
			settings=self.settings,
			logger=self.logger,
			opt_type=self.settings.opt_type,
			optimizer_state=self.optimizer_state,
			beta=self.beta,
			gamma=self.gamma,
			r=self.kdloss_wt,
			d=self.advloss_wt,
			eps=self.eps,
			step_size=self.step_size,
			steps=self.steps,
			adv_loss =self.adv_loss,
			)

	def run(self):
		best_top1 = 0
		rob_at_std_best = 0
		robust_best = 0
		std_at_robust_best = 0
		best_epoch = 0
		start_time = time.time()
  
		train_acc_list = []
		test_acc_list = []
		train_adv_acc_list = []
		test_adv_acc_list = []
		# train_f1_list = []
		# test_f1_list = []

		init_natural, init_robust, _ = self.trainer.test_teacher(0) #check teacher performance

		try:
			for epoch in tqdm(range(self.start_epoch, self.settings.nEpochs)):
				self.epoch = epoch
				self.start_epoch = 0

				train_error, train_adv_acc = self.trainer.train(epoch=epoch)
				test_acc, adv_acc = self.trainer.adv_test(epoch=epoch)

				#save results
				train_acc_list.append(100.0 - train_error)
				train_adv_acc_list.append(train_adv_acc)
				test_acc_list.append(test_acc)
				test_adv_acc_list.append(adv_acc)
    
				#output to csv
				with open(os.path.join(self.settings.save_path,"acc.csv"), "+a") as f:
					csvwriter = csv.writer(f, delimiter=',')
					csvwriter.writerow([100.0 - train_error, train_adv_acc, test_acc ,adv_acc])

				if best_top1 <= test_acc:
					best_top1 = test_acc
					rob_at_std_best = adv_acc
					tqdm.write("save best model")
					# self.trainer.save_models(epoch)
					
				#save every epoch
				self.trainer.save_current_models(epoch)
				if adv_acc >= robust_best:
					robust_best = adv_acc
					std_at_robust_best = test_acc
					best_epoch = epoch
					tqdm.write("save robust model")
					self.trainer.save_models(epoch, "robust_")	
     

				tqdm.write("#==>Standard Best is: Std. Accuracy: {:f}%, Robust Accuracy: {:f}%".format(best_top1, rob_at_std_best))
				tqdm.write("#==>Robust Best is: Std. Accuracy: {:f}%, Robust Accuracy {:f}%\n".format(std_at_robust_best, robust_best))

				self.logger.debug("#==>Best Result is: Std. Accuracy: {:f}%, Robust Accuracy: {:f}%".format(best_top1, rob_at_std_best))
				self.logger.debug("#==>Robust Best is: Std. Accuracy: {:f}%, Robust Accuracy {:f}%".format(std_at_robust_best, robust_best))


			message = self.settings.experimentID+"\n"
			# message += "#==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1, 100 - best_top5)
			message += "#==>Best Result is: Std. Accuracy: {:f}, Robust Accuracy: {:f}\n".format(std_at_robust_best, robust_best)

		except BaseException as e:
			self.logger.error("Training is terminating due to exception: {}".format(str(e)))
			message = self.settings.save_path+"\n"
			message += "Training is terminating due to exception: {}".format(str(e))
			requests.post(token,json={'text': message})
			traceback.print_exc()
		
		end_time = time.time()
		time_interval = end_time - start_time
		t_string = "Running Time is: " + str(datetime.timedelta(seconds=time_interval)) + "\n"
		self.logger.info(t_string)

		return best_top1, None, rob_at_std_best, robust_best, std_at_robust_best


def main():
	parser = argparse.ArgumentParser(description='DataFreeShield')
	parser.add_argument('--conf_path', type=str, metavar='conf_path',
	                    help='input the path of config file')
	parser.add_argument('--id', type=int, default=123, metavar='experiment_id', #for now, it's only used to set randomseed
	                    help='Experiment ID')
	parser.add_argument('--exp_name', type=str, default="default", 
	help='upper directory where this set of experiments will be saved')
	parser.add_argument('--gpu', type=str, metavar='gpu_id',
	                    help='GPU ID', default="0")

	#--------------------train
	parser.add_argument('--model', type=str, default='resnet20')

	parser.add_argument('--advloss', type=str)
 
	parser.add_argument('--advloss_wt', type=float, default=1.0,
	help="Hyperparameter to control Adversarial Loss. 0.0 deactivates adversarial training")
	parser.add_argument('--kdloss_wt', type=float, default=1.0,
	help="Hyperparameter to control Distillation Loss. 0.0 deactivates kd loss=(CE(S(x)) + KL(S(x),T(x))")
	parser.add_argument('--beta', type=float, default=1.0)
	parser.add_argument('--gamma', type=float, default=1.0)

	parser.add_argument("--inner_max", type=str, choices=["ce", "kl"], default="kl")
	
	#--------------------eval
	parser.add_argument('--eps', type=float, default=4, help='Epsilon for Adversarial Training')
	parser.add_argument('--steps', type=int, default=10)
	parser.add_argument('--step_size',type=float, default=1)
	parser.add_argument('--train_eps', type=float, default=4, help='Epsilon for Adversarial Training')
	parser.add_argument('--train_step_size',type=float, default=1)
	parser.add_argument('--train_steps', type=int, default=10)

	parser.add_argument('--norm', type=str, default='l_inf', choices=['l_inf', 'l_2'])
 
	#------------------record
	parser.add_argument('--csv', type=str, default=None,
	help='path to csv where results will be accumulated')

	#------------------optional
	parser.add_argument('--lr_S', type=float, default=None,
	help='Default is 0.0001. Raise if training from scratch')
	parser.add_argument('--teacher_label', action='store_true', help="Use Teacher Label as Ground Truth")
	parser.add_argument('--teacher_wt', type=str, default=None,
	help='path to pretrained weight')
	parser.add_argument('--data_num', type=int, default=60000,
	help="number of training data to use")
	parser.add_argument("--data_pth", type=str, default=None, nargs='+', help="dataset path to override original")
	parser.add_argument("--epoch", type=int, default=None)
	parser.add_argument("--multigpu", action='store_true')
	parser.add_argument("--etc", type=str, help="DB arg for storing any additional information")
	
	parser.add_argument('--p_thresh', type=float, default=0.0, help="agreement threshold. 0 means all params will be updated.")
	parser.add_argument('--agg_iter', type=int , default=1, help="number of batches to be aggregated for agreement scoring. 1 means standard training.")

	parser.add_argument('--batchSize', default=None, type=int)
	parser.add_argument('--from_scratch', action='store_true')
	

	args = parser.parse_args()
	
	option = Option(args.conf_path, args)
	option.manualSeed = args.id + 1

	command_line = ' '.join(sys.argv)
	option.set_save_path(command_line)

	experiment = ExperimentDesign(option)
	nat_best, _, rob_at_nat_best, rob, nat_at_rob = experiment.run()

	if option.csv is not None: #if save is enabled
		log(option.csv, 
		[args.conf_path, option.dataPath,
   		option.dataset+"("+str(option.use_real)+")", str(option.data_num),
		option.model, #"scratch("+str(option.from_scratch)+")",
		option.advloss, 
		str(option.kdloss_wt), str(option.advloss_wt), str(option.beta), str(option.gamma),
		option.attack, str(option.eps * 255), str(option.steps), str(option.step_size * 255),
		str(nat_best),str(rob_at_nat_best), str(nat_at_rob), str(rob)])


def log(name, contents):
	with open(name, 'a+') as f:
		csvwriter = csv.writer(f, delimiter=',')
		csvwriter.writerow(contents)

if __name__ == '__main__':
	main()

