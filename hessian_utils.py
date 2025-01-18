import torch
# import torchvision
# #from torch.utils.data import DataLoader

import torch.nn as nn
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

def compute_hessian(model: nn.Module, loss_fn, data_points, labels=None):
    """
    Compute the Hessian matrix of the loss with respect to model parameters.
    
    Args:
        model (nn.Module): The neural network model
        loss_fn: Loss function that takes model output (and optionally labels) as input
        data_points (torch.Tensor): Input data points
        labels (torch.Tensor, optional): Labels for supervised learning tasks
    
    Returns:
        torch.Tensor: Hessian matrix of shape (num_parameters, num_parameters)
    """
    # Get model parameters and their total count
    parameters = list(model.parameters())
    n_params = sum(p.numel() for p in parameters)
    
    # Flatten parameters for easier handling
    def get_params_vector():
        return torch.cat([p.flatten() for p in parameters])
    
    # Function to compute loss and gradients
    def get_loss_and_grad():
        # Forward pass
        outputs = model(data_points)
        
        # Compute loss
        if labels is not None:
            loss = loss_fn(outputs, labels)
        else:
            loss = loss_fn(outputs)
            
        # Compute gradients
        gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        grad_vector = torch.cat([g.flatten() for g in gradients])
        
        return loss, grad_vector
    
    # Initialize Hessian matrix
    hessian = torch.zeros(n_params, n_params)
    
    # Compute loss and first-order gradients
    loss, gradients = get_loss_and_grad()
    
    # Compute second-order derivatives
    for i in range(n_params):
        # Compute derivative of i-th gradient component with respect to all parameters
        second_derivs = torch.autograd.grad(gradients[i], parameters, retain_graph=True)
        second_derivs_vector = torch.cat([g.flatten() for g in second_derivs])
        hessian[i] = second_derivs_vector
        
    return hessian

def plot_hessian(model: nn.Module, loss_fn, data_points, labels=None):
    """
    Compute and plot the Hessian matrix of the loss with respect to model parameters.
    
    Args:
        model (nn.Module): The neural network model
        loss_fn: Loss function that takes model output (and optionally labels) as input
        data_points (torch.Tensor): Input data points
        labels (torch.Tensor, optional): Labels for supervised learning tasks
    """
    hessian_matrix = compute_hessian(model, loss_fn, data_points, labels)
    
    # Plot the Hessian matrix
    plt.imshow(hessian_matrix.detach().cpu().numpy(), cmap='viridis', aspect='auto', vmax=0.01, vmin=-0.01)
    plt.colorbar()
    
    # Add vertical and horizontal lines indicating layer weights
    param_sizes = [p.numel() for p in model.parameters()]
    param_cumsum = torch.cumsum(torch.tensor(param_sizes), dim=0).numpy()
    
    for pos in param_cumsum[:-1]:
        plt.axvline(x=pos - 0.5, color='red', linestyle='--')
        plt.axhline(y=pos - 0.5, color='red', linestyle='--')
    
    plt.title('Hessian Matrix')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Index')
    plt.show()


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
		# Normalize the direction
		direction = [d / torch.norm(d) for d in direction]
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
		all_grads = self.get_classwise_gradient_dot(direction)
		for i in range(self.n_classes):
			axes[i].plot(epsilons, perturbed_class_losses_tensor[:, i].numpy() - original_class_losses[i].numpy())
			axes[i].set_title(f'Class {i}')
			axes[i].set_xlabel('Alpha')
			axes[i].set_ylabel('Loss Change')
			axes[i].set_ylim(-0.1, 0.1)  # Set the same scale across all plots
			# get a straight vertical line at x = 0
			axes[i].axvline(x=0, color='red', linestyle='--')
			# get the slope of the tangent line at x = 0
			# get the tangent line at x = 0
			axes[i].plot(epsilons, all_grads[i] * epsilons, color='green', linestyle='--')


		plt.tight_layout()
		plt.show()
  
	def get_gradient_vector_dot(self, v):
		self.model.zero_grad()
		outputs = self.model(self.images)
		loss = self.criterion(outputs, self.labels)
		loss.backward(retain_graph=True)
		dot = torch.tensor(0.0)
		for p, v_ in zip(self.model.parameters(), v):
			dot += torch.dot(p.grad.reshape(-1), v_.reshape(-1))
		return dot
	
	def get_classwise_gradient_dot(self, v):
		self.model.zero_grad()
		outputs = self.model(self.images)
		dots = []
		for i in range(self.n_classes):
			self.model.zero_grad()
			class_mask = (self.labels == i)
			if class_mask.sum() > 0:
				loss = self.criterion(outputs[class_mask], self.labels[class_mask])
				loss.backward(retain_graph=True)
				dot = torch.tensor(0.0)
				for p, v_ in zip(self.model.parameters(), v):
					dot += torch.dot(p.grad.reshape(-1), v_.reshape(-1))
				dots.append(dot.item())
				#print(f'Class {i}: {dot.item()}')
			else:
				print(f'Class {i} has no samples')
				dots.append(0)
		return dots