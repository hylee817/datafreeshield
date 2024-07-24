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


# class DeepInversionTinyImageNet(Dataset):
#     def __init__(
#             self,
#             root:str,
#             data:torch.tensor=None,
#             targets:list=None,
#             train:bool=True,
#             transform:Optional[Callable]=None,
#             target_transform:Optional[Callable]=None,
#             download: bool=False,
#     ) -> None:
#         super(DeepInversionTinyImageNet, self).__init__(root, train, transform, target_transform, download)
#         self.data = data
#         self.targets = targets

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         img, target = self.data[index], self.targets[index]

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

class DeepInversionTinyImageNet(Dataset):
	"""Face Landmarks dataset."""

	def __init__(
				self, 
				root:str,
				data: torch.tensor=None,
				targets:list=None,
				transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.transform = transform
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# if torch.is_tensor(idx):
		#     idx = idx.tolist()

		# img_name = os.path.join(self.root_dir,
		#                         self.landmarks_frame.iloc[idx, 0])
		# image = io.imread(img_name)
		# landmarks = self.landmarks_frame.iloc[idx, 1:]
		# landmarks = np.array([landmarks])
		# landmarks = landmarks.astype('float').reshape(-1, 2)
		# sample = {'image': image, 'landmarks': landmarks}
		img, target = self.data[idx], self.targets[idx]

		if self.transform:
			img = self.transform(img)

		return img, target
	

class DeepInversionImageNet(Dataset):
	"""Face Landmarks dataset."""

	def __init__(
				self, 
				root:str,
				data: torch.tensor=None,
				targets:list=None,
				transform=None):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.transform = transform
		self.data = data
		self.targets = targets

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# if torch.is_tensor(idx):
		#     idx = idx.tolist()

		# img_name = os.path.join(self.root_dir,
		#                         self.landmarks_frame.iloc[idx, 0])
		# image = io.imread(img_name)
		# landmarks = self.landmarks_frame.iloc[idx, 1:]
		# landmarks = np.array([landmarks])
		# landmarks = landmarks.astype('float').reshape(-1, 2)
		# sample = {'image': image, 'landmarks': landmarks}
		img, target = self.data[idx], self.targets[idx]

		if self.transform:
			img = self.transform(img)

		return img, target