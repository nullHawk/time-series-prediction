# Time Series Prediction with LSTM

This project demonstrates a time-series prediction model for predicting a sine wave using PyTorch. It employs an LSTM-based architecture to learn and predict the behavior of a sine wave, showcasing the capability of recurrent neural networks for sequence modeling.

## Features
- Generates synthetic sine wave data for training and testing.
- Trains an LSTM-based model to predict future values of the sine wave.
- Visualizes predictions after each training step.
- Saves predictions as `.pdf` files for review.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

Install the required packages using:
```bash
pip install torch numpy matplotlib
```

## Usage

### 1. Generate Data
The sine wave data is generated with added random shifts. The `generate_sinewave.py` script provides:
- `generate()`: Creates training and testing data.
- `plot()`: Visualizes the sine wave.

### 2. Train the Model
Run `train.py` to train the LSTM model:
```bash
python train.py
```

During training:
- The model learns the sine wave pattern.
- Predictions for unseen data are plotted and saved at each training step (e.g., `predict1.pdf`, `predict2.pdf`, etc.).

### 3. Model Architecture
The `LSTMPredictor` class in `model.py` defines a custom LSTM model:
- **Two LSTM Cells**: For capturing complex temporal patterns.
- **Linear Layer**: Maps LSTM outputs to the target space.

### Results
Below is an example visualization of the model's predictions:

![image](https://github.com/user-attachments/assets/d9bb39f4-1cda-4ef0-b870-0eae21f7e032)


- **Solid Line**: Actual sine wave.
- **Dashed Line**: Predicted future values.

The model progressively learns the pattern as training progresses.

## File Structure
```
├── train.py               # Main script to train the model
├── model.py               # LSTM model definition
├── generate_sinewave.py   # Data generation and plotting
├── predictX.pdf           # Generated prediction plots (after each training step)
```

## Example Output
Training logs:
```
Step: 0
Loss: 0.0456
Test loss: 0.0312
Step: 1
Loss: 0.0213
Test loss: 0.0187
...
```

Plots (`predictX.pdf`) show how the model's predictions improve over steps.

---

Feel free to clone this repository and experiment with time-series prediction using LSTM!
