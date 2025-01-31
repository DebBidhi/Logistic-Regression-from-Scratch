# Classification using Logistic Regression from Scratch (without Neural Network)

## Project Overview
This project implements a logistic regression model from scratch to classify images as cars or non-cars using Python and NumPy, without using deep neural networks.

## Model Visualization

### Model's Car Perception
![image](https://github.com/user-attachments/assets/b53660ea-7708-422a-a499-428c1cfd38d6)
this is what my model's reconstruction of its learned parameters of **what is a car!**

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

## Requirements
- NumPy
- Matplotlib
- Pillow (PIL)
- SciPy

## Installation
```bash
pip install numpy matplotlib pillow scipy pickle
```
## Prediction Usage Example
1. Make sure the input image is a .png
2. Download just the **parameters.pkl** file
3. Set the image path in the code and Run predict_from_png(path)
   
```
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

with open('parameters.pkl', 'rb') as f:
    p = pickle.load(f)

def predict_from_png(image_path, w=p["w"], b=p["b"]):
    # Load and preprocess image
    img = Image.open(image_path)
    img = img.resize((72, 72))
    img_array = np.array(img) / 255.0
    
    Y_prediction = sigmoid(np.dot(w.T, img_array.reshape(72 * 72 * 4, 1)) + b)
    
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    isCar = Y_prediction > 0.5
    print("Prediction:", "Car" if isCar else "Non-Car")
    print(f"Car Probability: {Y_prediction[0][0]*100:.2f}%")
    
    return Y_prediction

car_image_path = 'path/to/car_image.png'
non_car_image_path = 'path/to/non_car_image.png'

predict_from_png(car_image_path)
predict_from_png(non_car_image_path)

```
## Example Output
![image](https://github.com/user-attachments/assets/5b9acb32-51c7-48e5-b900-3515be1099a7)

## Usage
1. Prepare your dataset
2. Run the logistic regression script
3. Analyze model performance and visualizations

## License
MIT license

## Acknowledgments
- Kaggle Car Non-Car Dataset


