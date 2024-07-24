"""
data loder for loading data
"""
import os
import math

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import struct
from collections import Counter
import utils.data_configs as data_configs
import medmnist
from medmnist import INFO


__all__ = ["DataLoader", "PartDataLoader"]


class DataLoader(object):
	
	def __init__(self, config=None):
		self.config = config

		## set data config
		dname = self.config.dataset
		real = True
		if "deep_inversion" in self.config.dataset:
			dname = self.config.dataset.replace("deep_inversion_","")
			real = False
		if "mnist" in self.config.dataset:
			subclass = dname
			dname = "medmnist"
		self.data_config = getattr(data_configs, f"{dname}_cfg")() #fetch class instance
		self.data_config.real = real
		if "mnist" in self.config.dataset:
			self.data_config.subclass = subclass
			self.data_config.info = INFO[subclass]
			self.data_config.n_classes = len(self.data_config.info['label'])

		## load dataset
		kwargs = {}
		if self.data_config.alias == "medmnist": 
			kwargs.update({'subclass':self.data_config.subclass})

		train_dataset , test_dataset = self.data_config.load(**kwargs)
		if not self.data_config.real:
			train_dataset = self.load_synthetic()

		self.train_len = len(train_dataset)
		self.test_len = len(test_dataset)
		self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												  batch_size=self.config.batchSize,
												  shuffle=True,
												  pin_memory=True,
												  num_workers=self.config.nThreads,
												)
		self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
												  batch_size=self.config.batchSize,
												  shuffle=False,
												  pin_memory=True,
												  num_workers=self.config.nThreads)
	
	def getloader(self):
		"""
		get train_loader and test_loader
		"""
		return self.train_loader, self.test_loader

	def get_len(self):
		return self.test_len

	def load_synthetic(self):
  
		train_dataset_list = []
		single_dataset_num = self.config.data_num//len(self.config.dataPath)
		for data_root in self.config.dataPath:
			data_root = data_root.replace("model_name", self.config.model)
			if self.config.dc: train_data_root = data_root.replace("datasets", "scratch")
			else: train_data_root = data_root
			single_dataset = torch.load(train_data_root)

			if single_dataset_num < len(single_dataset): #slice
				single_dataset.data = single_dataset.data[:single_dataset_num]
				single_dataset.targets = single_dataset.targets[:single_dataset_num]
			train_dataset_list.append(single_dataset)
			print("extracting {} from {}".format(len(single_dataset.data),data_root))
		train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)

		return train_dataset

	

