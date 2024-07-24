import torch
from typing import Any, Callable, Optional, Tuple
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class DeepInversionCIFAR10(datasets.CIFAR10):
	def __init__(
			self,
			root:str,
			data:torch.tensor=None,
			targets:list=None,
			train:bool=True,
			transform:Optional[Callable]=None,
			target_transform:Optional[Callable]=None,
			download: bool=False,
	) -> None:
		super(DeepInversionCIFAR10, self).__init__(root, train, transform, target_transform, download)
		self.data = data
		self.targets = targets
	
	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		img, target = self.data[index], self.targets[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

class DeepInversionSVHN(datasets.CIFAR10):
	def __init__(
			self,
			root:str,
			data:torch.tensor=None,
			targets:list=None,
			train:bool=True,
			transform:Optional[Callable]=None,
			target_transform:Optional[Callable]=None,
			download: bool=False,
	) -> None:
		super(DeepInversionSVHN, self).__init__(root, train, transform, target_transform, download)
		self.data = data
		self.targets = targets

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		img, target = self.data[index], self.targets[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

class DeepInversionCIFAR100(datasets.CIFAR100):
    def __init__(
            self,
            root:str,
            data:torch.tensor=None,
            targets:list=None,
            train:bool=True,
            transform:Optional[Callable]=None,
            target_transform:Optional[Callable]=None,
            download: bool=False,
    ) -> None:
        super(DeepInversionCIFAR100, self).__init__(root, train, transform, target_transform, download)
        self.data = data
        self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
