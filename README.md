# FGSM Adversarial Attack Demonstration

A full-stack application demonstrating the Fast Gradient Sign Method (FGSM) adversarial attack on an MNIST handwritten digit classifier.

## Project Overview

This project implements a complete pipeline for testing adversarial robustness:
*   **Backend:** A FastAPI service running a custom PyTorch Convolutional Neural Network trained on the MNIST dataset. It exposes an endpoint to perform FGSM attacks on uploaded images.
*   **Frontend:** A Next.js single-page application that provides an interactive interface for users to upload digits, adjust the perturbation strength (epsilon), and visualize the adversarial results in real-time.

## Deployed Services

The application has been successfully deployed and is accessible at the following locations:
*   **Frontend Web Application:** [Deployed on Vercel]
*   **Backend API Service:** [Deployed on Render]

## Local Installation and Execution

### Prerequisites
*   Python 3.10 or higher
*   Node.js 18 or higher
*   npm (Node Package Manager)

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\Activate.ps1
   # On Linux/MacOS:
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the FastAPI server:
   ```bash
   uvicorn app_fgsm:app --reload
   ```
   *Note: If the pre-trained model weights are not found, the server will automatically download the dataset and train a new model upon startup. The API will be available at http://127.0.0.1:8000. Documentation is available at http://127.0.0.1:8000/docs.*

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the required Node modules:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```
   *The web application will be accessible at http://localhost:3000.*

## Technical Explanation of FGSM

The Fast Gradient Sign Method (FGSM), introduced by Goodfellow et al. in 2015, is an effective adversarial attack technique that exploits how neural networks compute gradients during backpropagation. Instead of using gradients to update model weights to minimize loss, FGSM uses the gradient of the loss with respect to the input image to maximize the loss. 

The perturbation is calculated as the sign of this gradient, scaled by a factor of epsilon, and added to the original image. The fundamental equation is: x_adv = x + epsilon * sign(gradient_x(Loss)).

The key insight is that neural networks are often linearly sensitive in high-dimensional spaces. Consequently, making tiny but carefully directed changes to many pixels simultaneously can push the model's decision past a classification boundary. The epsilon parameter controls the magnitude of the perturbation. Larger values create more visible noise but are more likely to fool the model, while smaller values preserve visual quality but may not successfully change the prediction. This trade-off is central to evaluating adversarial robustness.

## Evaluation Observations

An automated evaluation script was run on 1,000 MNIST test samples to measure the attack success rate across various epsilon values.

| Epsilon | Clean Accuracy | Adversarial Accuracy | Accuracy Drop | Attack Success Rate |
|---------|---------------|---------------------|---------------|-------------------|
| 0.00    | 97.90%        | 97.90%              | 0.00%         | 0.00%             |
| 0.05    | 97.90%        | 94.10%              | 3.80%         | 3.80%             |
| 0.10    | 97.90%        | 85.20%              | 12.70%        | 12.70%            |
| 0.15    | 97.90%        | 69.70%              | 28.20%        | 28.20%            |
| 0.20    | 97.90%        | 47.90%              | 50.00%        | 50.00%            |
| 0.25    | 97.90%        | 24.40%              | 73.50%        | 73.50%            |
| 0.30    | 97.90%        | 9.80%               | 88.10%        | 88.10%            |

Key findings from the evaluation:
*   The baseline Convolutional Neural Network achieves 97.9% accuracy on clean test samples.
*   Even a minimal epsilon of 0.05 induces an accuracy loss of approximately 4%.
*   At an epsilon of 0.20, the attack successfully compromises the model on exactly half of all evaluated samples.
*   At an epsilon of 0.30, the model retains only a 9.8% accuracy rate, reflecting a massive 88% overall accuracy drop.
*   The relationship between epsilon and accuracy degradation is roughly linear up to an epsilon of 0.20. Beyond this point, the curve flattens as the model reaches near-total failure.
*   These results consistently demonstrate that increasing the perturbation magnitude strictly increases the attack success rate.

## Project Structure

```text
DevNeuronAssessment/
├── backend/
│   ├── app_fgsm.py          # FastAPI application handling API requests
│   ├── fgsm.py              # Implementation of the FGSM adversarial attack
│   ├── model.py             # Simple CNN architecture definition
│   ├── train.py             # Script for training the CNN on MNIST
│   ├── test_fgsm.py         # Utility script for testing the attack logic
│   ├── evaluate_fgsm.py     # Script to generate the evaluation metrics table
│   └── requirements.txt     # Python environment dependencies
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root Next.js layout with font setup
│   │   ├── page.tsx         # Main interactive interface component
│   │   └── globals.css      # Custom styling and design system
│   └── package.json         # Node.js project configuration
└── README.md                # Project documentation and setup guide
```

## References

*   Goodfellow, I.J., Shlens, J., and Szegedy, C. (2015). Explaining and Harnessing Adversarial Examples. International Conference on Learning Representations (ICLR).
*   PyTorch Documentation: FGSM Tutorial.
