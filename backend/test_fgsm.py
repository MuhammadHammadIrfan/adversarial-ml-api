import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SimpleCNN
from fgsm import FGSMAttack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

attack = FGSMAttack(model, device)

epsilon = 0.2

data_iter = iter(test_loader)
image, label = next(data_iter)

# Clean prediction
output = model(image.to(device))
init_pred = output.argmax(dim=1)

# Generate adversarial
perturbed_image = attack.generate(image, label, epsilon)

# Adversarial prediction
output_adv = model(perturbed_image)
final_pred = output_adv.argmax(dim=1)

print("Original:", init_pred.item())
print("Adversarial:", final_pred.item())