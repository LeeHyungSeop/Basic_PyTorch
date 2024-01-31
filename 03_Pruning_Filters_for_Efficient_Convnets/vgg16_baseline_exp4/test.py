import torch
import sys
sys.path.append("..")
from architecture2 import VGG16_BN
from preprocessing import val_loader
 
 
model = VGG16_BN()
model.load_state_dict(torch.load('./checkpoint/best_model.pth')['model_state_dict'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Top-1 Accuracy
correct = 0
total = 0
val_acc = 0.0
model.eval()
with torch.no_grad():
    for data in val_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

val_acc = (correct / total) * 100
print(f"Top-1 Accuracy : {val_acc:.5f}%")