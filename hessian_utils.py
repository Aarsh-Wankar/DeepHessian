import torch
# import torchvision
# #from torch.utils.data import DataLoader

# import torch.nn as nn
# import torch.optim as optim
#import torchvision.transforms as transforms

from PyHessian.pyhessian import hessian
from matplotlib import pyplot as plt


def save_model(model, path):
	torch.save(model.state_dict(), path)
 
def load_model(model, path):
	model.load_state_dict(torch.load(path))
	
def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        m_perb.data = m_orig.data + alpha * d
    return model_perb



def get_classwise_loss(model,criterion, images, labels, n_classes):
	class_losses = torch.zeros(n_classes)
	class_counts = torch.zeros(n_classes)
	model.eval()

	with torch.no_grad():
		outputs = model(images)
		for i in range(n_classes):
			class_mask = (labels == i)
			if class_mask.sum() > 0:
				class_loss = criterion(outputs[class_mask], labels[class_mask])
				class_losses[i] += class_loss.item() 
				class_counts[i] += class_mask.sum().item()
	#class_losses /= class_counts
	return class_losses

def coagulate_dataloader(dataloader):
	images = []
	labels = []
	for i, (image, label) in enumerate(dataloader):
		images.append(image)
		labels.append(label)
	images = torch.cat(images)
	labels = torch.cat(labels)
	return images, labels

class Hessian_model:
	def __init__(self, model, dataloader, loss, n_classes=10):
		self.model = model
		self.criterion = loss
		self.n_classes = n_classes
		self.dataloader= dataloader
		self.images, self.labels = coagulate_dataloader(dataloader)
		self.hessian_comp = hessian(model, self.criterion, dataloader=dataloader, cuda=False)

	def get_hessian(self):
		return self.hessian_comp

	def get_eigenvalues_eigenvectors(self, n=10):
		return self.hessian_comp.eigenvalues(top_n=n)

	def get_perturbed_model(self, epsilon, direction, model_perb):
		model_perb = get_params(self.model, model_perb, direction, epsilon)
		return model_perb
		
	def get_classwise_loss(self):
		return get_classwise_loss(self.model, self.criterion, self.images, self.labels, self.n_classes)

	def perturb_and_plot(self, direction, epsilons, base_model, title=None):
		perturbed_class_losses_list = []
		
		model_perb = base_model  
		original_class_losses = self.get_classwise_loss()
		for e in epsilons:
			model_perb = self.get_perturbed_model(e, direction, model_perb)
			perturbed_class_losses = get_classwise_loss(model_perb, self.criterion, self.images, self.labels, self.n_classes)
			perturbed_class_losses_list.append(perturbed_class_losses)
		perturbed_class_losses_tensor = torch.stack(perturbed_class_losses_list)

		fig, axes = plt.subplots(2, 5, figsize=(20, 10))
		if title:
			fig.suptitle(title)
		axes = axes.flatten()

		for i in range(10):
			axes[i].plot(epsilons, perturbed_class_losses_tensor[:, i].numpy() - original_class_losses[i].numpy())
			axes[i].set_title(f'Class {i}')
			axes[i].set_xlabel('Alpha')
			axes[i].set_ylabel('Loss Change')
			axes[i].set_ylim(-0.1, 0.1)  # Set the same scale across all plots

		plt.tight_layout()
		plt.show()
  
	def get_gradient_vector_dot(self, v):
		self.model.zero_grad()
		outputs = self.model(self.images)
		loss = self.criterion(outputs, self.labels)
		loss.backward()
		dot = 0
		for p, v_ in zip(self.model.parameters(), v):
			dot += torch.dot(p.grad.reshape(-1), v_.reshape(-1))
		return dot