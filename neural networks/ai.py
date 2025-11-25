import torch
import torch.nn as nn
import torch.nn.functional as F 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class Model(nn.Module):
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

df = pd.read_csv(input("Enter path to CSV file: "))

# Remove ID and empty columns
df = df.drop(['ID', 'Unnamed: 15'], axis=1, errors='ignore')

# Fill NaN values
df = df.fillna('Unknown')

# Define target column - use Category instead of Vulnerability for better results
# Vulnerability has 13,425 unique values (too many!)
# Category likely has much fewer classes
target_col = 'Category'
y = df[target_col]

# Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(f"Number of unique vulnerabilities: {len(label_encoder.classes_)}")
print(f"Total samples: {len(y)}")

# Get feature columns (everything except target)
X = df.drop(target_col, axis=1)

# Encode all categorical/text features to numeric
from sklearn.preprocessing import LabelEncoder as LE
X_encoded = pd.DataFrame()
for col in X.columns:
    le = LE()
    X_encoded[col] = le.fit_transform(X[col].astype(str))

# Split into train, validation, and test sets
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(X_encoded, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42)

# Normalize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#initialize model with correct dimensions
torch.manual_seed(45)
input_size = X_train.shape[1]
print(f"\nInput size: {input_size}")
num_classes = len(label_encoder.classes_)
model = Model(input_size=input_size, hidden_size=512, num_classes=num_classes)

#convert X and y to torch tensors
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)

# Create DataLoaders for mini-batch training
from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#set criterion and optimizer with label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
# Better scheduler - cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

#train the model with early stopping
num_epochs = 200
best_val_acc = 0
patience = 30
patience_counter = 0
loss_history = []
val_acc_history = []

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
        # Save best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Val Acc: {val_acc*100:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Early stopping
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
print(f'\nBest validation accuracy: {best_val_acc*100:.2f}%')

#graph it out
import matplotlib.pyplot as plt
plt.plot(range(1, len(loss_history)+1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    total = y_test.size(0)
    correct = (predicted == y_test).sum().item()
    
    print(f'\n=== Test Results ===')
    print(f'Total test samples: {total}')
    print(f'Correct predictions: {correct}')
    print(f'Incorrect predictions: {total - correct}')
    print(f'Accuracy: {100 * correct / total:.2f}%')

#save the trained model
torch.save(model.state_dict(), 'cybersecurity_model.pth')

#load the trained model
new_model = Model()
new_model.load_state_dict(torch.load('cybersecurity_model.pth'))