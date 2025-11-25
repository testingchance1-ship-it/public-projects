#!/usr/bin/env python3
"""
Generic CSV Classification Neural Network
==========================================
A flexible neural network trainer for CSV-based classification tasks.

Usage:
    python3 generic_learning.py

Requirements:
    - CSV file with labeled data
    - Target column must be specified during runtime
    - Automatically handles text and categorical features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


class NeuralNetworkClassifier(nn.Module):
    """
    4-layer feedforward neural network with batch normalization and dropout.
    Suitable for tabular classification tasks.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(0.4)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def load_and_preprocess_data(csv_path, target_column, columns_to_drop=None, 
                              test_size=0.15, val_size=0.18, random_state=42):
    """
    Load CSV and prepare data for training.
    
    Args:
        csv_path: Path to CSV file
        target_column: Name of the column to predict
        columns_to_drop: List of columns to exclude (e.g., IDs, irrelevant columns)
        test_size: Proportion of data for testing (default 0.15 = 15%)
        val_size: Proportion of remaining data for validation (default 0.18)
        random_state: Random seed for reproducibility
        
    Returns:
        Processed train, validation, and test sets, along with metadata
    """
    print(f"\n{'='*60}")
    print(f"Loading data from: {csv_path}")
    print(f"{'='*60}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"✗ Error: File '{csv_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)
    
    # Display basic info
    print(f"\nColumns in dataset: {list(df.columns)}")
    
    # Check if target column exists
    if target_column not in df.columns:
        print(f"\n✗ Error: Target column '{target_column}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Drop specified columns
    if columns_to_drop:
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        if existing_cols:
            df = df.drop(existing_cols, axis=1)
            print(f"✓ Dropped columns: {existing_cols}")
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        df = df.fillna('Unknown')
        print(f"✓ Filled {missing_before} missing values with 'Unknown'")
    
    # Separate features and target
    y = df[target_column]
    X = df.drop(target_column, axis=1)
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    num_classes = len(label_encoder.classes_)
    print(f"\n{'='*60}")
    print(f"Target column: '{target_column}'")
    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(y_encoded)}")
    
    # Check if we have too many classes
    if num_classes > 1000:
        print(f"⚠ WARNING: {num_classes} classes detected - this may be too many!")
        print(f"⚠ Consider using a column with fewer unique values")
        avg_samples = len(y_encoded) / num_classes
        print(f"⚠ Average samples per class: {avg_samples:.2f}")
    
    # Show class distribution summary
    class_counts = pd.Series(y_encoded).value_counts()
    print(f"Class distribution:")
    print(f"  - Most common class: {class_counts.iloc[0]} samples")
    print(f"  - Least common class: {class_counts.iloc[-1]} samples")
    print(f"  - Average per class: {len(y_encoded)/num_classes:.1f} samples")
    print(f"{'='*60}\n")
    
    # Encode all categorical features
    X_encoded = pd.DataFrame()
    print("Encoding features...")
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
    print(f"✓ Encoded {len(X.columns)} feature columns")
    
    # Split data: train (70%), val (15%), test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_encoded, y_encoded, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )
    
    print(f"✓ Data split complete:")
    print(f"  - Training: {len(X_train)} samples")
    print(f"  - Validation: {len(X_val)} samples")
    print(f"  - Test: {len(X_test)} samples")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print(f"✓ Features normalized\n")
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'num_classes': num_classes,
        'label_encoder': label_encoder,
        'input_size': X_train.shape[1]
    }


def train_model(data, hidden_size=512, num_epochs=200, batch_size=64, 
                learning_rate=0.002, patience=30):
    """
    Train neural network on prepared data.
    
    Args:
        data: Dictionary from load_and_preprocess_data()
        hidden_size: Number of neurons in hidden layers
        num_epochs: Maximum training epochs
        batch_size: Mini-batch size for training
        learning_rate: Initial learning rate
        patience: Early stopping patience (epochs without improvement)
        
    Returns:
        Trained model and training history
    """
    # Convert to tensors
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.LongTensor(data['y_train'])
    X_val = torch.FloatTensor(data['X_val'])
    y_val = torch.LongTensor(data['y_val'])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.LongTensor(data['y_test'])
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    torch.manual_seed(42)
    model = NeuralNetworkClassifier(
        input_size=data['input_size'],
        hidden_size=hidden_size,
        num_classes=data['num_classes']
    )
    
    print(f"{'='*60}")
    print(f"Model Architecture:")
    print(f"  - Input size: {data['input_size']}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Output classes: {data['num_classes']}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    loss_history = []
    val_acc_history = []
    
    print(f"{'='*60}")
    print(f"Training started (max {num_epochs} epochs, early stop patience={patience})")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val).sum().item() / len(y_val)
            val_acc_history.append(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:>3}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Val Acc: {val_acc*100:>5.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    print(f'\n{"="*60}')
    print(f'Best validation accuracy: {best_val_acc*100:.2f}%')
    print(f'{"="*60}\n')
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        total = y_test.size(0)
        correct = (predicted == y_test).sum().item()
    
    print(f'{"="*60}')
    print(f'Test Results:')
    print(f'{"="*60}')
    print(f'Total test samples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Incorrect predictions: {total - correct}')
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    print(f'{"="*60}\n')
    
    return {
        'model': model,
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'best_val_acc': best_val_acc,
        'test_accuracy': 100 * correct / total
    }


def plot_training_history(loss_history, val_acc_history):
    """Plot training loss and validation accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(range(1, len(loss_history)+1), loss_history, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy plot
    ax2.plot(range(1, len(val_acc_history)+1), 
             [acc*100 for acc in val_acc_history], 'g-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print(" Generic CSV Classification Neural Network")
    print("="*60)
    
    # Get CSV path
    csv_path = input("\nEnter path to CSV file: ").strip()
    
    # Get target column
    target_col = input("Enter target column name (column to predict): ").strip()
    
    # Optional: columns to drop
    drop_input = input("Enter columns to drop (comma-separated, or press Enter to skip): ").strip()
    columns_to_drop = [col.strip() for col in drop_input.split(',')] if drop_input else None
    
    # Load and preprocess data
    data = load_and_preprocess_data(csv_path, target_col, columns_to_drop)
    
    # Ask for training parameters
    print(f"\n{'='*60}")
    print("Training Configuration (press Enter for defaults)")
    print(f"{'='*60}")
    
    hidden_input = input("Hidden layer size [default: 512]: ").strip()
    hidden_size = int(hidden_input) if hidden_input else 512
    
    epochs_input = input("Max epochs [default: 200]: ").strip()
    num_epochs = int(epochs_input) if epochs_input else 200
    
    batch_input = input("Batch size [default: 64]: ").strip()
    batch_size = int(batch_input) if batch_input else 64
    
    lr_input = input("Learning rate [default: 0.002]: ").strip()
    learning_rate = float(lr_input) if lr_input else 0.002
    
    print()
    
    # Train model
    results = train_model(
        data,
        hidden_size=hidden_size,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Plot results
    plot_input = input("Show training plots? (y/n) [default: y]: ").strip().lower()
    if plot_input != 'n':
        plot_training_history(results['loss_history'], results['val_acc_history'])
    
    print("\n✓ Training complete! Best model saved to 'best_model.pth'\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
