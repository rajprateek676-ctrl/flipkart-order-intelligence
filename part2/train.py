
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"fashion_mnist"
SAMPLES=ROOT/"data"/"sample_images"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

tfm=transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])
raw=datasets.FashionMNIST(DATA,train=True,download=True)
labels=np.array(raw.targets)
idx=np.arange(len(raw))
tr_idx, val_idx=train_test_split(idx,test_size=5000,stratify=labels,random_state=42)

train_ds=datasets.FashionMNIST(DATA,train=True,transform=tfm)
test_ds=datasets.FashionMNIST(DATA,train=False,transform=tfm)
tr=Subset(train_ds,tr_idx); va=Subset(train_ds,val_idx)
train_loader=DataLoader(tr,batch_size=128,shuffle=True,num_workers=0)
val_loader=DataLoader(va,batch_size=256,num_workers=0)
test_loader=DataLoader(test_ds,batch_size=256,num_workers=0)

weights=models.ResNet18_Weights.DEFAULT
model=models.resnet18(weights=weights)
for p in model.parameters(): p.requires_grad=False
model.fc=nn.Linear(model.fc.in_features,10)
model.to(device)
opt=torch.optim.Adam(model.fc.parameters(),lr=1e-3)
loss_fn=nn.CrossEntropyLoss()

def accuracy(loader):
    model.eval(); correct=total=0
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            correct += (model(x).argmax(1)==y).sum().item()
            total += len(y)
    return correct/total

for epoch in range(5):
    model.train()
    for x,y in train_loader:
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        loss=loss_fn(model(x),y)
        loss.backward()
        opt.step()
    print(f"epoch={epoch+1} validation_accuracy={accuracy(val_loader):.4f}")

# Export real test images from raw Fashion-MNIST split.
SAMPLES.mkdir(parents=True,exist_ok=True)
for i in range(5):
    arr=np.array(raw) if False else np.array(datasets.FashionMNIST(DATA,train=False,download=False)[i][0])
    label=int(datasets.FashionMNIST(DATA,train=False,download=False)[i][1])
    Image.fromarray(arr.astype("uint8"),mode="L").save(SAMPLES/f"{i+1:02d}_{classes[label].replace('/','_').replace(' ','_')}.png")

torch.save(model.state_dict(),ROOT/"models"/"product_classifier.pt")
torch.save(model.state_dict(),ROOT/"part2"/"models"/"product_classifier.pt")
(ROOT/"part2"/"class_names.json").write_text(__import__("json").dumps(classes,indent=2))
print("saved model and 5 real test PNG samples")
