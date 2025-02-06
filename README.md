# Sentiment Analysis with LSTM and Transformer Models

## Overview
This repository contains two implementations of sentiment analysis using deep learning:
1. **LSTM-based model** (achieved 82.56% accuracy on test data)
2. **Transformer-based model** (achieved 80.91% accuracy on test data)

The models were trained on a sentiment dataset to classify text into positive, negative, or neutral sentiments.

## Dataset
- The dataset consists of labeled text data for sentiment analysis.
- Preprocessing includes tokenization, padding, and vectorization.

## Models
### 1. LSTM Model
- Uses an embedding layer followed by LSTM layers.
- Achieved an accuracy of **82.56%** on the test set.
- Model architecture:
  - Input Layer: Sequence length 120
  - Embedding Layer: 64-dimensional
  - Spatial Dropout
  - LSTM Layer: 128 units
  - Dropout Layer
  - Fully Connected Layer: 10 neurons
  - Output Layer: 1 neuron
- Total Parameters: **740,117**

### 2. Transformer Model
- Utilizes a transformer-based architecture for text classification.
- Achieved an accuracy of **80.91%** on the test set.
- Model architecture:
  - Input Layer: Sequence length 120
  - Token and Position Embedding Layer: 32-dimensional
  - Transformer Block
  - Global Average Pooling Layer
  - Dropout Layers
  - Fully Connected Layers (20 neurons, then 1 neuron output)
- Total Parameters: **655,177**

## Training
- Models were trained using TensorFlow/Keras and PyTorch.
- Hyperparameters:
  - Batch Size: 32
  - Learning Rate: 0.001
  - Epochs: 20 (early stopping at epoch 9 for Transformer, 17 for LSTM)
- Loss function: Cross-Entropy Loss
- Optimizer: Adam

## Results
| Model         | Accuracy | Precision | Recall | F1 Score |
|--------------|---------|-----------|--------|----------|
| LSTM         | 82.56%  | 82.80%    | 82.13% | 82.46%   |
| Transformer  | 80.91%  | 81.18%    | 80.40% | 80.79%   |
