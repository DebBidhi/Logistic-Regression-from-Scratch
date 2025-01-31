# Classification using Logistic Regression from Scratch (without Neural Network)

## Project Overview
This project implements a logistic regression model from scratch to classify images as cars or non-cars using Python and NumPy, without using deep neural networks.

## Model Visualization

### Model's Car Perception
**This is what my model's reconstruction of its learned parameters of what is a car!**

![image](https://github.com/user-attachments/assets/b8483ac2-67e5-45b0-b86e-2e339cf08897)

The above image is reconstructed with the ``visualize_learned_car()`` function that is implemented within the notebook. The purpose of this is to see how the model imagines a car from all the learning, and it's incredible to think it's learned all this within ~10 seconds of training time on google collab. ``Time taken for model training: 10.768696546554565 seconds``

And also, it's not even a Neural Network!!

## Example Output
![image](https://github.com/user-attachments/assets/50fa8d0c-23fc-47f6-b125-96f1c7fb170e)
![image](https://github.com/user-attachments/assets/0266b8b3-6f44-450d-bc1d-2df9946640c5)

## Requirements
- NumPy
- Matplotlib
- Pillow (PIL)
- SciPy
- pickle

## Installation
```bash
pip install numpy matplotlib pillow scipy pickle
```
## Prediction Usage Example
1. Make sure the input image is a .png
2. Download just the **parameters.pkl** file
3. Set the image path in the code and Run predict_from_png(path)
   
```
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pickle


with open('parameters.pkl', 'rb') as f:
    p = pickle.load(f)

def sigmoid(z):
    """
    Compute the sigmoid of z
    """
    return 1 / (1 + np.exp(-z))

def predict_from_png(image_path, w = p["w"], b = p["b"]):
    """
    Given any png image path it predicts whether it's a car or not

    Arguments:
    image_path -- path to the image
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """

    img = Image.open(image_path)
    img = img.resize((72, 72))
    img_array = np.array(img)

    img_array = img_array / 255.0

    Y_prediction = sigmoid(np.dot(w.T, img_array.reshape(72 * 72 * 4, 1)) + b)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis('off')

    isCar = Y_prediction > 0.5
    
    Pred_Probablity = Y_prediction[0][0] * 100
    
    prediction_text = "Prediction: Car" if isCar else "Prediction: Not a Car"
    probability_text = f"Probability: {Pred_Probablity:.5f}%"
    
    plt.text(10, 20, prediction_text, color='white', 
             fontsize=12, fontweight='bold', 
             bbox=dict(facecolor='green' if isCar else 'red', alpha=0.7))
    plt.text(10, 50, probability_text, color='white', 
             fontsize=10, 
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.show()

car_image_path = 'path/to/car_image.png'
non_car_image_path = 'path/to/non_car_image.png'

predict_from_png(car_image_path)
predict_from_png(non_car_image_path)

```

## Key Features
- Built entirely using NumPy
- Implemented core logistic regression components from scratch
- Image preprocessing and feature extraction
- Hyperparameter tuning with learning rate analysis
- Model performance visualization

## Dataset
- Car and non-car images from Kaggle's Car Non-Car dataset
- Image size: 72x72 pixels
- Preprocessed and normalized for training

## Technical Implementation
- Sigmoid activation function
- Gradient descent optimization
- Binary classification with 0.5 threshold

## Model Performance
- Training Accuracy: 99%
- Testing Accuracy: 93-95%

## Learning Curves
![image](https://github.com/user-attachments/assets/12f4bef7-0840-44ef-917c-17b932c37acf)

## Function Representation
![image](https://github.com/user-attachments/assets/f2beeebd-6d60-40d5-8787-741ba96d1588)

## Usage
1. Prepare your dataset
2. Run the logistic regression NoteBook
3. Analyze model performance and visualizations

## License
MIT license

## Acknowledgments
- Kaggle Car Non-Car Dataset


