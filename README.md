# Autism Classification using Vision Transformer (ViT)

## Overview
This project focuses on classifying autism using **Vision Transformer (ViT)**. The model was fine-tuned on a labeled autism dataset for binary classification. Hugging Face was used for efficient model training and evaluation.

## Key Features
- **Vision Transformer (ViT)**: Fine-tuned pre-trained model for autism classification.
- **Binary Classification**: Labels include `Autism` and `Non-Autism`.
- **Performance Metrics**:
  - Accuracy: 85.5%
  - F1-Score: 85.5%

## Technologies Used
- Python
- Hugging Face
- transformers
- Wandb
- Pytorch
- Matplotlib

## Dataset
The dataset contains images used for autism classification, split into training, validation, and test sets:

- **Training**: 70%
- **Validation**: 20%
- **Testing**: 10%

## Results
- **Model**: Vision Transformer (ViT) fine-tuned for binary classification.
- **Performance**:
  - Test Accuracy: 85.5%
  - F1-Score: 85.5%

## File Descriptions
- **`Autism-Classification-ViT.ipynb`**: Jupyter notebook for training the ViT model.
- **`inference.py`**: Script for running inference on new images.


## Instructions to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Autism-Classification-ViT.git
cd Autism-Classification-ViT
