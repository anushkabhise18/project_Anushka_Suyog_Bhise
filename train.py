import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets
from config import epochs, batch_size, data_dir, learning_rate, checkpoint_dir, checkpoint_path, patience
from config import train_ratio, val_ratio, test_ratio, random_seed
from model import ASLClassifier
from dataset import ASLDataset, get_transforms, create_dataloaders

def train_model(model, num_epochs, train_loader, val_loader=None, test_loader=None, learning_rate=learning_rate, patience=patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model = model.to(device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    
    best_val_acc = 0.0
    no_improve_epochs = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        start_time = time.time()

        # Training loop
        train_losses = []
        train_correct = 0
        train_total = 0
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            train_total += y.size(0)

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        if val_loader:
            val_losses = []
            val_correct = 0
            val_total = 0
            model.eval()

            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    pred = model(x)
                    loss = loss_fn(pred, y)

                    val_losses.append(loss.item())
                    val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                    val_total += y.size(0)

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_acc = val_correct / val_total
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(avg_val_acc)

            scheduler.step(avg_val_loss)

            # Save model if validation accuracy improved
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path} (validation accuracy: {best_val_acc:.4f})")
                no_improve_epochs = 0 
            else:
                no_improve_epochs += 1  
            # Early stopping check
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

            print(f"Epoch: {epoch+1}/{num_epochs} - {time.time()-start_time:.1f}s - "
                  f"Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc:.4f} - "
                  f"Val Loss: {avg_val_loss:.4f} - Val Acc: {avg_val_acc:.4f}")
        else:
            print(f"Epoch: {epoch+1}/{num_epochs} - {time.time()-start_time:.1f}s - "
                  f"Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc:.4f}")

    
    if not val_loader:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # Test
    if test_loader:
        print("\nEvaluating on test set...")
        model.eval()
        test_losses = []
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                test_losses.append(loss.item())
                test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                test_total += y.size(0)

        avg_test_loss = sum(test_losses) / len(test_losses)
        avg_test_acc = test_correct / test_total

        print(f"Test Loss: {avg_test_loss:.4f} - Test Acc: {avg_test_acc:.4f}")

    # Plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if val_loader:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if val_loader:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    return model, history

def train_and_evaluate():
    print(f"Total samples in dataset: {len(datasets.ImageFolder(root=data_dir))}")
    print(f"Number of classes: {len(datasets.ImageFolder(root=data_dir).classes)}")
    print(f"Classes: {datasets.ImageFolder(root=data_dir).classes}")
    
    print("Using data augmentation only on training set")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_dir=data_dir,
        batch_size=batch_size,
        train_transform=train_transform,
        val_transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed
    )
    
    model = ASLClassifier()
    model, history = train_model(
        model, epochs, train_loader, val_loader, test_loader,
        learning_rate=learning_rate,
        patience=patience
    )
    
    return model, history


if __name__ == "__main__":
    train_and_evaluate()