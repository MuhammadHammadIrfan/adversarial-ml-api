import io
import base64
from typing import Annotated

import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from fgsm import FGSMAttack
from model import SimpleCNN

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="FGSM Adversarial Attack API",
    description="Demonstrates the Fast Gradient Sign Method (FGSM) attack on an MNIST CNN model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model & attack initialization
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

attack = FGSMAttack(model, device)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),  # Invert: MNIST is white-on-black
])


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------
class AttackResponse(BaseModel):
    clean_prediction: int
    adversarial_prediction: int
    attack_success: bool
    adversarial_image: str  # base64-encoded PNG


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "FGSM Adversarial Attack API is running"}


@app.post("/attack", response_model=AttackResponse)
def run_attack(
    file: Annotated[UploadFile, File(description="Image file (PNG/JPEG)")],
    epsilon: Annotated[float, Form(description="Perturbation magnitude")] = 0.1,
) -> AttackResponse:
    """
    Run an FGSM adversarial attack on the uploaded image.

    1. Classifies the clean image.
    2. Generates an adversarial example using FGSM with the given epsilon.
    3. Classifies the adversarial image.
    4. Returns both predictions, the attack success flag, and the adversarial
       image as a base64-encoded PNG.
    """
    # Read & preprocess -------------------------------------------------
    contents = file.file.read()
    image = Image.open(io.BytesIO(contents))
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Clean prediction --------------------------------------------------
    with torch.no_grad():
        output = model(image_tensor)
        clean_pred = output.argmax(dim=1)

    # Generate adversarial image ----------------------------------------
    perturbed_image = attack.generate(image_tensor.clone(), clean_pred, epsilon)

    # Adversarial prediction --------------------------------------------
    with torch.no_grad():
        output_adv = model(perturbed_image)
        adv_pred = output_adv.argmax(dim=1)

    # Encode adversarial image as base64 --------------------------------
    adv_img = perturbed_image.squeeze().detach().cpu()
    adv_img_pil = transforms.ToPILImage()(adv_img)
    buffer = io.BytesIO()
    adv_img_pil.save(buffer, format="PNG")
    adv_img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return AttackResponse(
        clean_prediction=int(clean_pred.item()),
        adversarial_prediction=int(adv_pred.item()),
        attack_success=bool(clean_pred.item() != adv_pred.item()),
        adversarial_image=adv_img_base64,
    )