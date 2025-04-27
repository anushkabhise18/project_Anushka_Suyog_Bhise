import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from model import ASLClassifier  # Change to your model class name if different
from config import img_height, img_width, norm_mean, norm_std, checkpoint_path, class_names

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ASLPredictDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), img_path

def predict_asl(list_of_img_paths):
    """
    Takes a list of image file paths (from the data/ directory) and returns a list of predicted class labels.
    """
    # Load model and weights
    model = ASLClassifier().to(device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model weights not found at {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()


    idx_to_class = {i: name for i, name in enumerate(class_names)}

    dataset = ASLPredictDataset(list_of_img_paths)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    predictions = []
    with torch.no_grad():
        for batch, _ in loader:
            batch = batch.to(device)
            outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend([class_names[p.item()] for p in preds])

    return predictions
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