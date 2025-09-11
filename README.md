# SimpleCTM 

![CTM Diagram](diagram.png)

A super-simplified, single-file PyTorch replica of the **Continuous Thought Machine (CTM)** from Sakana AI. 

## TL;DR

Modern AI often ignores the timing and synchronization in biological brains for efficiency. CTM bridges that gap, using neural dynamics as the core of computation. This simplified version distills the essence of the [original repo](https://github.com/SakanaAI/continuous-thought-machines) into one Python file (`ctm.py`), training on MNIST to classify digits while demonstrating key CTM concepts like neuron synchronization and temporal thinking.

Inspired by the original work: [Continuous Thought Machines](https://pub.sakana.ai/ctm/) and [Sakana AI](http://sakana.ai/ctm/).

## Installation 🛠️

1. Clone the repo:
   ```bash
   git clone https://github.com/xandykati98/SimpleCTM.git
   cd SimpleCTM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Requires PyTorch, Torchvision – check `requirements.txt` for details)

## Usage 🎮

Run the training script:
```bash
python ctm.py
```

This will:
- Download MNIST
- Train the SimplifiedCTM model for 10 epochs
- Print progress and accuracy
- Save the model to `ctm_model.pth`

Customize hyperparameters in `main()` – like number of neurons, max ticks, epochs.

Example output:
```
Using device: cpu
=== Model Information ===
Total trainable parameters: 276,052
...
Epoch 1/10 completed - Average Loss: 2.3026, Accuracy: 11.24%
...
Model saved to ctm_model.pth
```

## How It Works 🧠

- **Image Encoding**: Flattens and reduces MNIST images to a 4D vector.
- **Internal Ticks**: Loops over time steps, updating pre/post activations.
- **Synapse Model**: Connects neurons with image input.
- **Neuron-Level Models**: Each neuron processes its activation history.
- **Synchronization**: Computes weighted dot products between neuron pairs with learnable decay.
- **Attention**: Synchronizations query the encoded image.
- **Prediction**: Reads combined features for classification.
- **Early Stopping**: Stops thinking when confident (>80%) or max ticks reached.

For full details, dive into `ctm.py` or the original [Sakana AI CTM page](https://pub.sakana.ai/ctm/).

*Built with ❤️ by xandykati98*
