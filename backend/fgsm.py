import torch
import torch.nn as nn

class FGSMAttack:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, image, label, epsilon):
        
        image = image.to(self.device)
        label = label.to(self.device)

        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        loss = self.criterion(output, label)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Collect gradient
        data_grad = image.grad.data
        
        # Apply FGSM formula
        sign_data_grad = data_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        
        # Clamp
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        
        return perturbed_image