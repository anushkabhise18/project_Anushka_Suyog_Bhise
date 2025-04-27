# Training parameters
batch_size = 64
learning_rate = 1e-3
epochs = 10
num_classes = 29  # 26 letters + space + delete + nothing
patience = 3
num_workers = 4

# Image configurations
img_height = 200
img_width = 200
input_channels = 3

# Path configurations
data_dir = "./data" #path to your data directory 
checkpoint_dir = "./checkpoints"
checkpoint_path = "./checkpoints/final_weights.pth"

# Data splitting parameters
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 18

# Normalization parameters
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']