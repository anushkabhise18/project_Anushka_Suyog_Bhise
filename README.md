# ASL Alphabet Recognition

## Project Overview
Sign language is an essential mode of communication for individuals with hearing and speech impairments.
However, there exists a communication barrier between sign language users and those who do not
understand it. Automating sign recognition can bridge this gap by enabling real-time translation of ASL into
text or speech. The goal of this project is to develop a convolutional neural network (CNN) model that
can recognize ASL signs from images and classify them accurately. Future plans after this would be to
convert this into speech and form sentences in real time.

### Dataset 
The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26
are for the letters A-Z and 3 classes for SPACE, DELETE and NOTHING.
- Input: A colour image of a hand showing an ASL sign (e.g., .jpg or .png format) Pre processing of
input before use, normalization, noise removal and contrast enhancement and
data augmentation.
- Output: A categorical label corresponding to the recognized ASL sign (e.g., ‘A’, ‘B’, ‘C’ ... ‘Z’ or
space, delete, nothing).

### Download Instructions
- Download from Kaggle: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- After downloading and unzipping, place the `asl_alphabet_train/` folder inside your project directory.

Optional (Colab):
```python
!pip install kagglehub
import kagglehub

path = kagglehub.dataset_download("grassknoted/asl-alphabet")
print("Dataset downloaded to:", path)
```
If you place the dataset elsewhere, update the `data_dir` variable in `config.py`.
By default, the data_dir has the path to the data folder, update as required.

# Cloning the Repository
The final_weights.pth is ~290 Mb and hence was uploaded using lfs.

To clone this repository along with large files managed by Git LFS, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/anushkabhise18/project_Anushka_Suyog_Bhise
    cd project_Anushka_Suyog_Bhise
    ```

2. Pull the large files managed by Git LFS:

    ```bash
    git lfs pull
    ```

This ensures that all large files (such as model weights) are properly downloaded.

> **Note:**  
> If you do not have Git LFS installed on your system, install it first:


## Model Architecture
The model is a Convolutional Neural Network (CNN) with the following architecture:

Input layer: Takes RGB images of size 200 × 200 × 3.
Convolutional blocks:
- Block 1: Conv2D (64 filters, 3×3 kernel, padding=1) → BatchNorm → ReLU → MaxPooling (2×2)
- Block 2: Conv2D (128 filters, 3×3 kernel, padding=1) → BatchNorm → ReLU → MaxPooling (2×2)
- Block 3: Conv2D (256 filters, 3×3 kernel, padding=1) → BatchNorm → ReLU → MaxPooling (2×2)
- Block 4: Conv2D (512 filters, 3×3 kernel, padding=1) → BatchNorm → ReLU → MaxPooling (2×2)

Classifier:
- Flatten layer
- Dropout (0.5)
- Dense layer (1024 units) → ReLU
- Dropout (0.5)
- Dense layer (num_classes units) (producing raw logits)

The model extracts low-level to high-level spatial features using multiple convolutional layers, applies batch normalization and ReLU activations for stable and non-linear learning, and uses max pooling for dimensionality reduction. The classifier part applies dropout regularization to prevent overfitting and maps the extracted features to the target class logits.



## Training Process
- **Data Splitting**: Training (70%), Validation (15%), Test (15%)
- **Data Augmentation**: Applied during training for better generalization
  - Random rotation (±10°)
  - Random horizontal flips
  - Color jitter (brightness, contrast)
- **Training Parameters**:
  - Batch size: 64
  - Learning rate: 0.001
  - Optimizer: Adam
  - Loss function: Cross-Entropy
  - Epochs: 10 with early stopping
  - Learning rate reduction on plateau

**Note:** Training accuracy may appear slightly lower than validation accuracy due to data augmentation applied only during training, which makes the training task more challenging.

## Installation and Setup
1. Clone this repository:
git clone https://github.com/yourusername/asl-recognition.git
cd asl-recognition

2. Install the required packages:
pip install torch torchvision matplotlib pillow numpy

3. Download the ASL Alphabet dataset from Kaggle and extract it to the `data` directory (or update `data_dir` in `config.py`)

## Usage

### Training
To train the model from scratch:
python train.py

This will:
1. Load and preprocess the ASL dataset
2. Create and train the CNN model
3. Save the trained model weights to `checkpoints/final_weights.pth`
4. Display training/validation curves

### Prediction
To run predictions on new images:
python predict.py --image_path path/to/image.jpg folder

Input- image

Output- Predicted label 

Example code to run predict using data folder-
```python
if __name__ == "__main__":
    
    data_dir_path = os.path.join(os.getcwd(), "./data")  # Path to YOUR data folder
    image_paths = [
        os.path.join(data_dir_path, f) 
        for f in os.listdir(data_dir_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    # Get predictions
    preds = predict_asl(image_paths)
    
    # Print results
    for path, pred in zip(image_paths, preds):
        print(f"{os.path.basename(path)}: {pred}")
```

## Future Improvements
- Implement real-time recognition through webcam input
- Extend the model to recognize dynamic gestures for complete words
- Improve robustness to different lighting conditions and backgrounds
- Create a mobile application for on-device inference


