import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ucimlrepo import fetch_ucirepo 

import warnings
warnings.simplefilter('ignore', FutureWarning)

# The [Breast Cancer Wisconsin (Diagnostic) dataset]
# (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) is a 
# classic dataset used for classification tasks. It contains 569 samples of breast cancer cells, 
# each with 30 features. The dataset is divided into two classes: benign and malignant. 
# The goal is to classify the breast cancer cells into one of the two classes.

breast_canser_repo = fetch_ucirepo(id=17)

X = breast_canser_repo.data.features
y = breast_canser_repo.data.targets

len(X.columns)


df = pd.concat([X, y], axis=1)
df.head()

y['Diagnosis'].value_counts()

# Diagnosis
# B    357
# M    212
# Name: count, dtype: int64

df.shape

# (569, 31) data is imbalanced. Lets randomly choose equal number of samples

df_B = df[df["Diagnosis"]=='B']
df_M = df[df["Diagnosis"]=='M']


df_B = df_B.sample(n=210, random_state=42)
df_M = df_M.sample(n=210, random_state=42)

df_balanced = pd.concat([df_B, df_M])
df_balanced.shape

# There are 210 samples in each class, with a total of 420 samples. 
# It means that the dataset is balanced.
# We will use 80% of the samples for training and 20% for testing.
# Lets split the data and prepare for training

X = df_balanced.drop("Diagnosis", axis=1)
y = df_balanced["Diagnosis"]

# # We need to convert catogorical variables to numbers
y = y.map({"B": 0, "M": 1})

num_features = len(X.columns)

# # Train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# # Standardizing data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Pytorch works with tensors. We need to convert our feature and target
# variable to tensors

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)



# To feed pytorch neural network we need dataloader objects

from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(X_train_tensor, y_train_tensor) 
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True)


# Define nn architecture specify loss and optimize


class ClassificationNet(torch.nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super(ClassificationNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_units, hidden_units)
        self.fc2 = torch.nn.Linear(hidden_units, output_units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = ClassificationNet(input_units=num_features, hidden_units=64, output_units=2)

print(model)


criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 10

train_losses = []
test_losses = []

train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0 
    total_train=0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct_train += (preds == y_batch).sum().item()
        total_train += y_batch.size(0)
        
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = correct_train /total_train
    train_acc_list.append(train_accuracy)

    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_test += (preds == y_batch).sum().item()
            total_test += y_batch.size(0)
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    test_accuracy = correct_test / total_test
    test_acc_list.append(test_accuracy)

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')

import matplotlib.pyplot as plt

# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# Plot train test accuracy curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_acc_list, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_acc_list, label='Test Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy Curve')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.show()

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import torch


def evaluate_model_performance(model, X_test_tensor, y_test_tensor, class_names=["Class 0", "Class 1"]):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
        preds = torch.argmax(outputs, dim=1).numpy()
        true = y_test_tensor.numpy()

    # Confusion Matrix
    cm = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(true, preds, target_names=class_names))

    # ROC Curve
    fpr, tpr, _ = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(true, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

evaluate_model_performance(model, X_test_tensor, y_test_tensor, class_names=["Benign", "Malignant"])
