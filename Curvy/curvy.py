import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn
import numpy as np
from typing import List, Optional, Callable, Tuple, Dict, Any, Iterator
import warnings

class Curvy(Optimizer):
    """
    Optimizer that periodically computes Hessian-based curvature and scales gradients
    before applying momentum updates.
    
    Args:
        params (iterable): Iterable of parameters to optimize
        lr (float): Learning rate
        momentum (float, optional): Momentum factor (default: 0.9)
        dampening (float, optional): Dampening for momentum (default: 0)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        nesterov (bool, optional): Enables Nesterov momentum (default: False)
        hessian_compute_interval (int): Number of iterations between Hessian computations
        hessian_n_iter (int): Number of iterations for Hessian computation
        hessian_epsilon (float): Epsilon for curvature scaling
        hessian_computer (callable): Function to compute Hessian
        criterion (callable): Loss function
        hessian_data_loader (torch.utils.data.DataLoader): DataLoader for Hessian computation
        cuda (bool): Whether to use CUDA
        cuda_device (str): CUDA device to use (default: "cuda:0")
    """
    
    def __init__(self, params, lr: float, momentum: float = 0.9, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False,
                 hessian_compute_interval: int = 100, hessian_n_iter: int = 20, n_hess_samples: int = 100,
                 hessian_epsilon: float = 1e-6, hessian_computer: Callable = None,
                 criterion: Callable = None, train_x: Any = None, train_y: Any = None, model: nn.Module = None,
                 cuda: bool = False, cuda_device = "cuda:0"):
        
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if hessian_compute_interval < 1:
            raise ValueError(f"Invalid hessian_compute_interval: {hessian_compute_interval}")
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        
        super(Curvy, self).__init__(params, defaults)
        
        self.hessian_compute_interval = hessian_compute_interval
        self.hessian_n_iter = hessian_n_iter
        self.hessian_epsilon = hessian_epsilon
        self.hessian_computer = hessian_computer
        self.criterion = criterion
        self.train_X = train_x
        self.train_y = train_y
        self.cuda = cuda
        if(self.cuda):
            if isinstance(cuda_device, torch.device):
                self.cuda_device = cuda_device
            else:
                self.cuda_device = torch.device(cuda_device)
        else:
            self.cuda_device = torch.device('cpu')
        self.iteration_count = 0
        self.mean_curvature_values = None
        self.n_hess_samples = n_hess_samples
        self.model = model
        
        if hessian_computer is None or criterion is None or train_x is None:
            print("No hessian computer")

            raise Exception(
                "Hessian computation function, criterion, or data loader not provided. "
                "Curvature-based scaling will be disabled."
            )
    
    def _compute_curvature(self):
        """Compute curvature based on Hessian."""
        if self.hessian_computer is None or self.criterion is None or self.train_X is None:
            raise Exception("Hessian Computer or Criterion or Training Data not provided.")

        # Store original training mode and set to eval for Hessian computation
        training_mode = self.model.training
        self.model.eval()
        
        # Compute Hessian and curvature
        indices = torch.randperm(self.train_X.size(0))[:self.n_hess_samples]
        random_hessian_loader = (self.train_X[indices], self.train_y[indices])
        
        # print(random_hessian_loader)
        hessian = self.hessian_computer(
            self.model, 
            self.criterion, 
            data=random_hessian_loader, 
            cuda=self.cuda,
            cuda_device=self.cuda_device
        )
        curvature = hessian.curvature_array(self.hessian_n_iter, 1e-3)
        mean_curvature = self._mean_curvature(curvature)
        
        # Restore original training mode
        self.model.train(training_mode)
        mx = max([c.abs().max() for c in mean_curvature])
        mn = min([c.abs().min() for c in mean_curvature])

        print("Max curvature:", mx.item())
        print("Min curvature:", mn.item())
        return mean_curvature
    
    # def _mean_curvature(self, curvature):
    #     """Compute mean curvature from curvature tensor."""
    #     return [torch.tensor(np.mean([curvature[i][j] for i in range(len(curvature))], axis=0)).to('cuda' if self.cuda else 'cpu') for j in range(len(curvature[0]))]
    def _mean_curvature(self, curvature):
        """Compute mean curvature from curvature tensor."""
        device = self.cuda_device if self.cuda else torch.device('cpu')
        return [
            torch.as_tensor(
                np.mean([curvature[i][j] for i in range(len(curvature))], axis=0),
                device=device
            )
            for j in range(len(curvature[0]))
        ]
    
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        self.iteration_count += 1
        
        # Check if we need to compute Hessian-based curvature
        if self.iteration_count % self.hessian_compute_interval == 0:
            # Extract model from parameters
            param_groups = self.param_groups
            if not param_groups:
                return loss
            
            # Assuming all parameters belong to the same model
            # This is a bit of a hack to get the model
            # for group in param_groups:
            #     for p in group['params']:
            #         if hasattr(p, '_orig_module'):
            #             model = p._orig_module
            #             break
            # else:
            #     # If we can't find the model, we'll have to skip curvature computation
            #     warnings.warn("Could not find model for Hessian computation")
            #     model = None
            
            # if self.model is not None:
            self.mean_curvature_values = self._compute_curvature()
        
        # Perform optimization step with curvature-based gradient scaling
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            param_index = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Apply weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                # Apply curvature-based scaling if available
                if self.mean_curvature_values is not None and param_index < len(self.mean_curvature_values):
                    cur = self.mean_curvature_values[param_index]
                    if cur is not None and cur.shape == d_p.shape:
                        # d_p = d_p / torch.max(
                        #     torch.abs(cur), 
                        #     torch.ones_like(cur) * self.hessian_epsilon
                        # )
                        # Move curvature to the same device as the grad
                        cur = cur.to(d_p.device, non_blocking=True)
                        eps = torch.full_like(cur, self.hessian_epsilon)
                        denom = torch.max(torch.abs(cur), eps)
                        d_p = d_p / denom
                param_index += 1
                
                # Apply momentum
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                
                # Update parameter
                p.data.add_(d_p, alpha=-group['lr'])
        
        return loss

    def zero_grad(self, set_to_none: bool = False):
        """Clears the gradients of all optimized parameters."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()


# Helper function to attach the model to parameters for later reference
def register_model_to_params(model):
    """
    Registers the model to its parameters to allow retrieval during optimization.
    
    Args:
        model (nn.Module): The model to register
    """
    for param in model.parameters():
        param._orig_module = model
        

# Example usage
def example_usage():
    """Example of how to use the optimizer."""
    import torch.nn as nn
    import torch.utils.data as data
    
    # Define a simple model
    model = nn.Linear(10, 2)
    criterion = nn.MSELoss()
    
    # Register model to parameters (needed for Hessian computation)
    register_model_to_params(model)
    
    # Create a dummy Hessian computer function
    def dummy_hessian_computer(model, criterion, data, cuda):
        class DummyHessian:
            def curvature_array(self, n_iter, epsilon):
                return [torch.ones_like(p) for p in model.parameters()]
        return DummyHessian()
    
    # Create dummy data loader
    dummy_dataset = data.TensorDataset(torch.randn(100, 10), torch.randn(100, 2))
    dummy_dataloader = data.DataLoader(dummy_dataset, batch_size=10)
    
    # Create optimizer
    optimizer = Curvy(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        hessian_compute_interval=5,
        hessian_n_iter=10,
        hessian_epsilon=1e-6,
        hessian_computer=dummy_hessian_computer,
        criterion=criterion,
        hessian_data_loader=dummy_dataloader,
        cuda=torch.cuda.is_available()
    )
    
    # Training loop
    for epoch in range(2):
        for inputs, targets in dummy_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    print("Training completed successfully!")