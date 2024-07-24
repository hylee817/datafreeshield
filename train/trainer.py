"""
basic trainer
"""
from cProfile import label
import time
import math

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
import os
from tqdm import tqdm
import copy
import collections
from PIL import Image
import csv
import psutil

from train.robustness_utils.dfshield import loss_dfshield
from robustness_utils.perturb import perturb_input

__all__ = ["Trainer"]

class Trainer(object):
	"""
	trainer for training network, use SGD
	"""
	
	def __init__(self, model, model_teacher, lr_master_S,
				train_loader, test_loader, test_len, 
				data_config, settings, logger,
				opt_type="SGD", optimizer_state=None,
				step_size=2/255, beta=6, gamma=1,eps=8/255, steps=10, r=1.0, d=1.0, 
				adv_loss=None):
		"""
		init trainer
		"""
		
		self.settings = settings

		self.p_thresh = self.settings.p_thresh
		
		self.model = utils.data_parallel( 
			model, self.settings.nGPU, self.settings.GPU)
		self.model_teacher = utils.data_parallel(
			model_teacher, self.settings.nGPU, self.settings.GPU)

		self.data_config = data_config

		self.train_loader = train_loader
		self.test_loader = test_loader

		self.lr_master_S = lr_master_S
		self.opt_type = opt_type
		if opt_type == "SGD":
			self.optimizer_S = torch.optim.SGD(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				momentum=self.settings.momentum,
				weight_decay=self.settings.weightDecay,
				nesterov=True,
			)
		elif opt_type == "RMSProp":
			self.optimizer_S = torch.optim.RMSprop(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1.0,
				weight_decay=self.settings.weightDecay,
				momentum=self.settings.momentum,
				alpha=self.settings.momentum
			)
		elif opt_type == "Adam":
			self.optimizer_S = torch.optim.Adam(
				params=self.model.parameters(),
				lr=self.lr_master_S.lr,
				eps=1e-5,
				weight_decay=self.settings.weightDecay
			)
		else:
			assert False, "invalid type: %d" % opt_type
		if optimizer_state is not None:
			self.optimizer_S.load_state_dict(optimizer_state)\

		self.logger = logger

		self.step_size = step_size
		self.eps = eps
		self.beta = beta
		self.gamma = gamma
		self.steps = steps
		self.adv_loss = adv_loss
		self.test_len = test_len
		self.r = r
		self.d = d

		self.train_eps = self.settings.train_eps
		self.train_step_size = self.settings.train_step_size

	def update_lr(self, epoch):
		"""
		update learning rate of optimizers
		:param epoch: current training epoch
		"""
		lr_S = self.lr_master_S.get_lr(epoch)
		# update learning rate of model optimizer
		for param_group in self.optimizer_S.param_groups:
			param_group['lr'] = lr_S
	

	def backward_S(self, loss_S):
		"""
		backward propagation
		"""
		self.optimizer_S.zero_grad()
		loss_S.backward()
		self.optimizer_S.step()
	
	def save_models(self,epoch,version=""):
		saved_flag = False
		while not saved_flag:
			try:
				torch.save(self.model.state_dict(),os.path.join(self.settings.save_path,version+f"model.pt"))
				saved_flag = True
			except OSError as e:
				waiting_flag = True
				while waiting_flag:
					available_space = psutil.disk_usage(self.settings.save_path).free/(2**(10*2)) ## MB 
					print(available_space, "MB left in storage")
					if available_space > 100:
						waiting_flag = False  # Storage is no longer full, exit the loop.
					time.sleep(30)  # Wait for 60 seconds before checking again.

	def save_current_models(self, epoch):
		saved_flag = False
		while not saved_flag:
			try:
				torch.save({'epoch':epoch,
							'optimizer_state_dict':self.optimizer_S.state_dict(),
							'model_state_dict':self.model.state_dict()}, 
							os.path.join(self.settings.save_path, "last_model.pth"))
				saved_flag = True
			except OSError as e:
				waiting_flag = True
				while waiting_flag:
					available_space = psutil.disk_usage(self.settings.save_path).free/(2**(10*2)) ## MB 
					print(available_space, "MB left in storage")
					if available_space > 100:
						waiting_flag = False  # Storage is no longer full, exit the loop.
					time.sleep(30)  # Wait for 60 seconds before checking again.

	def reset_(self, module):
		for p in module.parameters():
			p.data = p.data * 0
			p.requires_grad = False


	def train(self, epoch):

		"""
		training
		"""
		top1_error = utils.AverageMeter()
		top1_loss = utils.AverageMeter()
		t_acc = utils.AverageMeter()

		# predefine loss terms
		cross_entropy = nn.CrossEntropyLoss().cuda()
		kl_divergence = nn.KLDivLoss().cuda()

		correct = 0
		correct_adv = 0
		total_len = 0
		
		#TODO: tmp
		self.update_lr(epoch)

		self.model.train() #global model
		self.model_teacher.eval()
		
		agg_sign = copy.deepcopy(self.model).cuda()
		agg_pos = copy.deepcopy(self.model).cuda()
		agg_neg = copy.deepcopy(self.model).cuda()
		self.reset_(agg_sign)
		self.reset_(agg_pos)
		self.reset_(agg_neg)


		for i, (images, labels) in enumerate(self.train_loader):
			## refresh
			self.model.train()
			model_proxy = copy.deepcopy(self.model).cuda()
			# model_proxy.train()

			images, labels = images.cuda(), labels.cuda()    

			total_len += images.shape[0]

			output_t= self.model_teacher(images)
			if self.settings.teacher_label: #use teacher label as ground truth
				labels = output_t.max(1)[1]

			# forward student
			output = model_proxy(images)
			preds = output.max(1)[1]
			correct += len((preds == labels).nonzero())

			# knowledge distillation loss
			# for clean accuracy
			T = self.settings.temperature
			loss_KD = (T*T) * kl_divergence(F.log_softmax(output/T, dim=1), F.softmax(output_t.detach()/T, dim=1))

			if self.advloss == "DFShieldLoss":
				output_adv, loss_adv, loss_r, loss_t = loss_dfshield(model=model_proxy,
								x_natural=images,
								y=labels,
								teacher_outputs=output_t.detach(),
								optimizer=self.optimizer_S,
								step_size=self.train_step_size,
								epsilon=self.train_eps,
								perturb_steps=self.steps,
								beta=self.beta,
								gamma=self.gamma,
								temp=self.settings.temperature,
								distance=self.settings.norm,
								inner_max=self.settings.inner_max,
								) #returns: KL(S(x'),T(x)) + KL(S(x'), S(x))
				preds_adv = output_adv.max(1)[1]
				survivors = (preds_adv == labels).nonzero()
				correct_adv += len(survivors)
				loss_S = self.r * loss_KD + self.d * loss_adv

			else: #standard training (Knowledge Distillation)
				loss_CE = cross_entropy(output, labels)
				loss_S = loss_KD + loss_CE


			# update bn stats IMPORTANT 
			for m, mm in zip(self.model.modules(), model_proxy.modules()):
				if isinstance(m, nn.BatchNorm2d):
					m.running_mean = mm.running_mean
					m.running_var = mm.running_var

			##backward -> compute gradient
			loss_S.backward()

			# aggregate 
			for p, p_sign, p_pos, p_neg in zip(model_proxy.parameters(), agg_sign.parameters(), agg_pos.parameters(), agg_neg.parameters()):
				'''
				++ -> +2
				-- -> -2
				+- -> 0
				'''
				grad_sign = torch.sign(p.grad)
				p_sign.data += grad_sign

				pos_mask = 1 * (p.grad >= 0)
				neg_mask = 1 * (p.grad < 0)
				p_pos += (pos_mask * p.grad.data) #accumulate pos grad
				p_neg += (neg_mask * p.grad.data) #accumulate neg grad

			# actual update
			if ((i+1) % self.settings.agg_iter == 0) or ((i+1) == len(self.train_loader)):
				agreement_score = 0
				param_counter = 0
				self.optimizer_S.zero_grad() # grad <- None
				for p, p_sign, p_pos, p_neg in zip(self.model.parameters(), agg_sign.parameters(), agg_pos.parameters(), agg_neg.parameters()):
					update_mask = torch.abs(p_sign) >= (self.p_thresh * self.settings.agg_iter) #majority voted. update
					update_val = torch.where(p_sign >= 0, p_pos, p_neg)
					update_val = torch.where(p_sign == 0, p_pos + p_neg, update_val) #if tau==0, this will update using all gradients
					new_grad = update_mask * update_val
					p.grad = new_grad.clone()

					agreement_score += torch.sum(update_mask)
					param_counter += torch.sum(torch.ones(update_mask.shape))

				self.optimizer_S.step() # does not modify .grad
				self.optimizer_S.zero_grad() # grad <- None

				self.reset_(agg_sign)
				self.reset_(agg_pos)
				self.reset_(agg_neg)

			single_error, single_loss, single5_error = utils.compute_singlecrop(
				outputs=output, labels=labels,
				loss=loss_S, top5_flag=True, mean_flag=True)
			
			top1_error.update(single_error, images.size(0))
			top1_loss.update(single_loss, images.size(0))
			
			# end_time = time.time()
			
			gt = labels.data.cpu().numpy()
			d_acc = np.mean(np.argmax(output_t.data.cpu().numpy(), axis=1) == gt)

			t_acc.update(d_acc)


		top1_adv_acc = 100 * (correct_adv / total_len)
		tqdm.write(
			"[Epoch %d/%d] [Batch %d/%d] [Teacher acc: %.4f%%] [Natural acc: %.4f%% (%d/%d)] [Robust acc: %.4f%% (%d/%d)] [S loss: %f] [lr_S: %f] [Agreement: %.0f%%]"
			% (epoch + 1, self.settings.nEpochs, 
			i+1, len(self.train_loader), 
			100 * t_acc.avg, 
			100.0 - top1_error.avg, correct, total_len, 
			top1_adv_acc, correct_adv, total_len,
			loss_S.item(), 
			self.lr_master_S.get_lr(epoch), 
			# self.optimizer_S.param_groups[0]['lr']
			100 * (agreement_score / param_counter)
		))

		return top1_error.avg, top1_adv_acc
	
	def adv_test(self, epoch):
		"""
		test robustness
		"""

		self.model.eval()
		iters = len(self.test_loader)

		tot_init_correct = 0
		tot_attack_correct = 0

		# Loop over all examples in test set
		for i, data_pair in enumerate(self.test_loader):
			data, target = data_pair
			data, target = data.cuda(), target.cuda()

			# Set requires_grad attribute of tensor. Important for Attack
			data.requires_grad = True

			# Forward pass the data through the model
			output = self.model(data)
			init_pred = output.max(1)[1] # get the index of the max log-probability
			init_result = (init_pred == target)
			init_correct = init_result.nonzero() #indices of correct preds

			# Create adversarial examples
			perturbed_data = perturb_input(model=self.model,
										x_natural=data,
										y=target,
										step_size=self.step_size,
										epsilon=self.eps,
										perturb_steps=self.steps,
										distance=self.settings.norm
										)

			# Re-classify the perturbed image
			output = self.model(perturbed_data)

			# Check for success
			final_pred = output.max(1)[1] # get the index of the max log-probability
			attack_result = (final_pred == target)
			attack_correct = attack_result.nonzero()
			tot_init_correct += len(init_correct)
			tot_attack_correct += len(attack_correct)
		
		init_acc = tot_init_correct / float(self.test_len) * 100.0
		attack_acc = tot_attack_correct / float(self.test_len) * 100.0

		tqdm.write(
			"[Epoch %d/%d] [Batch %d/%d] [Test acc: %.4f%%(%d/%d)] [Robust acc: %.4f%%(%d/%d)]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, 
			init_acc, tot_init_correct, self.test_len,
			attack_acc, tot_attack_correct, self.test_len,
			)
		)

		return init_acc, attack_acc


	def test_teacher(self, epoch):
		"""
		test robustness
		# sanity checked: returns the same std. accuracy as original test code
		"""

		self.model_teacher.eval()
		iters = len(self.test_loader)

		tot_init_correct = 0
		tot_attack_correct = 0

		# Loop over all examples in test set
		for i, data_pair in enumerate(self.test_loader):
			data, target = data_pair
			data, target = data.cuda(), target.cuda()

			# Set requires_grad attribute of tensor. Important for Attack
			data.requires_grad = True

			# Forward pass the data through the model
			output = self.model_teacher(data)
			init_pred = output.max(1)[1] # get the index of the max log-probability
			init_result = (init_pred == target)
			init_correct = init_result.nonzero() #indices of correct preds

			# Create adversarial examples
			perturbed_data = perturb_input(model=self.model_teacher,
								x_natural=data,
								y=target,
								step_size=self.step_size,
								epsilon=self.eps,
								perturb_steps=self.steps,
								distance=self.settings.norm,
								)

			# Re-classify the perturbed image
			output = self.model_teacher(perturbed_data)

			# Check for success
			final_pred = output.max(1)[1] # get the index of the max log-probability
			attack_result = (final_pred == target)
			attack_correct = attack_result.nonzero()
			tot_init_correct += len(init_correct)
			tot_attack_correct += len(attack_correct)
		
		init_acc = tot_init_correct / float(self.test_len) * 100.0
		attack_acc = tot_attack_correct / float(self.test_len) * 100.0

		tqdm.write(
			"Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [robust acc: %.4f%%]"
			% (epoch + 1, self.settings.nEpochs, i + 1, iters, init_acc, attack_acc)
		)

		return init_acc, attack_acc, None
