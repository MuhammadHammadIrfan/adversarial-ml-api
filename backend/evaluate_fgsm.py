"""
Evaluate FGSM attack robustness across multiple epsilon values.

This script loads the pretrained MNIST CNN, applies the FGSM attack with
various epsilon values on 1000 test samples, and prints the accuracy drop.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fgsm import FGSMAttack
from model import SimpleCNN

# ── Config ────────────────────────────────────────────────────────────────
EPSILONS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
NUM_SAMPLES = 1000
BATCH_SIZE = 1

# ── Setup ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

attack = FGSMAttack(model, device)


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate(epsilon: float) -> dict:
    """Return accuracy and attack-success-rate for a given epsilon."""
    correct_clean = 0
    correct_adv = 0
    attack_success = 0
    total = 0

    for i, (image, label) in enumerate(test_loader):
        if i >= NUM_SAMPLES:
            break

        image, label = image.to(device), label.to(device)

        # Clean prediction
        with torch.no_grad():
            output = model(image)
            clean_pred = output.argmax(dim=1)

        if clean_pred.item() == label.item():
            correct_clean += 1

        # Adversarial prediction
        perturbed = attack.generate(image.clone(), label, epsilon)
        with torch.no_grad():
            output_adv = model(perturbed)
            adv_pred = output_adv.argmax(dim=1)

        if adv_pred.item() == label.item():
            correct_adv += 1

        if clean_pred.item() != adv_pred.item():
            attack_success += 1

        total += 1

    return {
        "epsilon": epsilon,
        "clean_accuracy": correct_clean / total * 100,
        "adversarial_accuracy": correct_adv / total * 100,
        "accuracy_drop": (correct_clean - correct_adv) / total * 100,
        "attack_success_rate": attack_success / total * 100,
    }


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("FGSM Attack Robustness Evaluation")
    print(f"Model: SimpleCNN  |  Dataset: MNIST (first {NUM_SAMPLES} test samples)")
    print("=" * 72)
    print(
        f"{'Epsilon':>8}  {'Clean Acc.':>11}  {'Adv. Acc.':>10}  "
        f"{'Acc. Drop':>10}  {'Attack Success':>15}"
    )
    print("-" * 72)

    for eps in EPSILONS:
        result = evaluate(eps)
        print(
            f"{result['epsilon']:>8.2f}  "
            f"{result['clean_accuracy']:>10.2f}%  "
            f"{result['adversarial_accuracy']:>9.2f}%  "
            f"{result['accuracy_drop']:>9.2f}%  "
            f"{result['attack_success_rate']:>14.2f}%"
        )

    print("=" * 72)
    print("Done. Take a screenshot of the above results for submission.")
