import os
import time
import shutil

from pyhocon import ConfigFactory

from utils.opt_static import NetOption
import random
import json

class Option(NetOption):
	def __init__(self, conf_path, args):
		super(Option, self).__init__()
		self.conf = ConfigFactory.parse_file(conf_path)
  
		#  ------------ General options ----------------------------------------
		self.save_path = self.conf['save_path']
		self.dataPath = self.conf['dataPath']  # path for loading data set
		if args.data_pth != None: #if specified, overwrite
			# self.dataPath = args.data_pth.split(" ")
			self.dataPath = args.data_pth
		self.dataset = self.conf['dataset']
		self.nGPU = self.conf['nGPU']  # number of GPUs to use by default
		self.GPU = self.conf['GPU']  # default gpu to use, options: range(nGPU)
		self.visible_devices = args.gpu
		
		# ------------- Data options -------------------------------------------
		self.nThreads = self.conf['nThreads']  # number of data loader threads
		self.nClasses = self.conf['nClasses']  # number of classes in the dataset
		
		# ---------- Optimization options --------------------------------------
		self.nEpochs = self.conf['nEpochs']  # number of total epochs to train
		if args.epoch != None:
			self.nEpochs = args.epoch
		self.batchSize = self.conf['batchSize']  # mini-batch size
		if args.batchSize != None:
			self.batchSize = args.batchSize
		self.momentum = self.conf['momentum']  # momentum
		self.weightDecay = float(self.conf['weightDecay'])  # weight decay
		self.opt_type = self.conf['opt_type']

		self.lr_S = self.conf['lr_S']  # initial learning rate
		if args.lr_S != None:
			self.lr_S = args.lr_S #overwrite if lr is given
		self.lrPolicy_S = self.conf['lrPolicy_S']  # options: multi_step | linear | exp | const | step
		self.step_S = self.conf['step_S']  # step for linear or exp learning rate policy
		self.decayRate_S = self.conf['decayRate_S']  # lr decay rate
		
		# ---------- Model options ---------------------------------------------
		self.model = args.model
		# self.from_scratch = self.conf['from_scratch']
		self.from_scratch = args.from_scratch
		self.freeze_bn = args.freeze_bn
	
		# ----------KD options ---------------------------------------------
		self.temperature = self.conf['temperature']
		
		# --------- Adversarial Train -----------------------------------
		self.norm = args.norm

		self.advloss = args.advloss
		self.advloss_wt = args.advloss_wt #hyperparameter to control adversarial loss
		self.kdloss_wt = args.kdloss_wt #hyperparameter to control kd loss
		self.beta = args.beta
		self.gamma = args.gamma

		self.train_eps = args.train_eps
		self.train_step_size = args.train_step_size
		self.train_steps = args.train_steps

		self.eps = args.eps
		self.steps = args.steps
		self.step_size = args.step_size

		self.inner_max = args.inner_max

		#--------------Additional options
		self.teacher_wt = args.teacher_wt #pretrained teacher weight
		self.teacher_label = args.teacher_label
		self.data_num = args.data_num
		self.csv = args.csv
		self.multigpu = args.multigpu
		self.etc = args.etc

		self.p_thresh = args.p_thresh
		self.agg_iter = args.agg_iter
		self.batchmean = args.batchmean
		# self.p_thresh_scaler = args.p_thresh_scaler
		# self.update_cond = args.update_cond

		identifier = ""				
		self.fullID = "{}_{}_{}_{}_{}_kd_{}_adv_{}_beta_{}_gamma_{}{}".format(
			self.data_num, self.model, self.advloss, 
			int(self.train_eps), 
			int(self.eps),
			self.kdloss_wt, self.advloss_wt, self.beta, self.gamma, identifier)
		self.experimentID = self.conf['experimentID'] + self.fullID
		self.exp_name = args.exp_name

  		#--------------------------
		self.eps = self.eps / 255. #rescale into 0-1 range
		self.step_size = self.step_size / 255.
		self.train_eps = self.train_eps / 255. #rescale into 0-1 range
		self.train_step_size = self.train_step_size / 255.
	
	def set_save_path(self, cmd):
		# structure: save_{{dataset}}/{{exp_name}}/{{experiment_ID}}
		self.exp_path = os.path.join(self.save_path, self.exp_name) 		# 2nd upper most directory
		self.save_path = os.path.join(self.exp_path, self.experimentID)

		flag = False
		repeat = -1
		save_path = self.save_path
		while not flag:
			try:
				os.makedirs(save_path)
				flag = True
				self.save_path = save_path
				print("Experiment will be saved at {}".format(self.save_path))
			except:
				repeat += 1
				save_path = self.save_path + "_repeat{}".format(repeat)
				time.sleep(random.uniform(0.1, 2.0))
	
		#backup
		with open(os.path.join(self.save_path,'command_line.txt'),'w') as f:
			f.write(cmd)

		cfg_json = dict()
		for vn, vv in vars(self).items():
			cfg_json.update({vn:vv})
		with open(os.path.join(self.save_path,'config.json'), 'w') as f:
			json.dump(cfg_json, f)
	
