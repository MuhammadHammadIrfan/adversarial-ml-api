# FGSM Adversarial Attack Demo

A full-stack application demonstrating the **Fast Gradient Sign Method (FGSM)** adversarial attack on an MNIST handwritten digit classifier.

| Component | Tech Stack |
|-----------|------------|
| Backend   | FastAPI, PyTorch, Pillow |
| Frontend  | Next.js 16, React 19, TypeScript |
| ML Model  | Custom CNN trained on MNIST |

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm

### Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Train the model — a pretrained mnist_cnn.pth is included
python train.py

# Run the server
uvicorn app_fgsm:app --reload
```

The API will be available at **http://127.0.0.1:8000**. Visit http://127.0.0.1:8000/docs for Swagger UI.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## Deployed URLs

| Service  | URL |
|----------|-----|
| Frontend | Deployed on **Vercel** |
| Backend  | Deployed on **Render** |

---

## Explanation of FGSM

The **Fast Gradient Sign Method (FGSM)**, introduced by Goodfellow et al. (2015), is a simple yet effective adversarial attack technique. It exploits the way neural networks compute gradients during backpropagation. Instead of using the gradient to update model weights (as in training), FGSM uses the gradient of the loss with respect to the **input image** to craft a small perturbation. The perturbation is computed as the sign of this gradient, scaled by a factor ε (epsilon), and then added to the original image: `x_adv = x + ε · sign(∇_x J(θ, x, y))`.

The key insight is that neural networks are often **linearly sensitive** in high-dimensional spaces — even tiny, carefully directed changes to many pixels simultaneously can push the model's decision past a classification boundary. The ε parameter controls the magnitude of the perturbation: larger values create more visible noise but are more likely to fool the model, while smaller values preserve visual quality but may not change the prediction. This trade-off is fundamental to understanding adversarial robustness.

---

## Observations

Evaluation results on 1,000 MNIST test samples:

| Epsilon | Clean Accuracy | Adversarial Accuracy | Accuracy Drop | Attack Success Rate |
|---------|---------------|---------------------|---------------|-------------------|
| 0.00    | 97.90%        | 97.90%              | 0.00%         | 0.00%             |
| 0.05    | 97.90%        | 94.10%              | 3.80%         | 3.80%             |
| 0.10    | 97.90%        | 85.20%              | 12.70%        | 12.70%            |
| 0.15    | 97.90%        | 69.70%              | 28.20%        | 28.20%            |
| 0.20    | 97.90%        | 47.90%              | 50.00%        | 50.00%            |
| 0.25    | 97.90%        | 24.40%              | 73.50%        | 73.50%            |
| 0.30    | 97.90%        | 9.80%               | 88.10%        | 88.10%            |

**Key findings:**
- The model achieves **97.9% clean accuracy** on the first 1,000 test samples.
- Even a small ε of **0.05** causes ~4% accuracy loss.
- At ε = **0.20**, the attack succeeds on **half** of all samples.
- At ε = **0.30**, only **9.8%** of predictions remain correct — an **88% accuracy drop**.
- Increasing epsilon monotonically increases attack success rate, confirming that larger perturbations produce stronger attacks.
- The relationship between ε and accuracy drop is roughly **linear** up to ε ≈ 0.20, then the curve flattens as the model is nearly fully compromised.

---

## Project Structure

```
DevNeuronAssessment/
├── backend/
│   ├── app_fgsm.py          # FastAPI application (POST /attack endpoint)
│   ├── fgsm.py              # FGSM attack class implementation
│   ├── model.py             # SimpleCNN model architecture
│   ├── train.py             # Model training script
│   ├── test_fgsm.py         # Quick FGSM test script
│   ├── evaluate_fgsm.py     # Full evaluation across epsilon values
│   ├── requirements.txt     # Python dependencies
│   └── mnist_cnn.pth        # Pretrained model weights
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root layout with fonts & metadata
│   │   ├── page.tsx         # Main SPA page (upload, slider, results)
│   │   └── globals.css      # Design system (dark theme, glassmorphism)
│   ├── package.json
│   └── next.config.ts
└── README.md
```

---

## References

- Goodfellow, I.J., Shlens, J., & Szegedy, C. (2015). [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572). *ICLR 2015*.
- [PyTorch FGSM Tutorial](https://pytorch.org/tutorials/beginner/fgsm_tutorial.html)
