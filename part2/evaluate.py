
import json, torch, numpy as np, pandas as pd
from pathlib import Path
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
ROOT=Path(__file__).resolve().parents[1]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes=json.loads((ROOT/"part2/class_names.json").read_text())
tfm=transforms.Compose([transforms.Grayscale(3),transforms.Resize((224,224)),transforms.ToTensor(),
                        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
ds=datasets.FashionMNIST(ROOT/"data/fashion_mnist",train=False,download=True,transform=tfm)
loader=DataLoader(ds,batch_size=256,num_workers=0)
m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,10)
m.load_state_dict(torch.load(ROOT/"models/product_classifier.pt",map_location=device)); m.to(device);m.eval()
ys=[];ps=[]
with torch.no_grad():
    for x,y in loader:
        p=m(x.to(device)).argmax(1).cpu().numpy()
        ys.extend(y.numpy());ps.extend(p)
cm=confusion_matrix(ys,ps)
report=classification_report(ys,ps,target_names=classes,digits=4)
(ROOT/"part2/confusion_matrix.txt").write_text("Classes: "+str(classes)+"\n\n"+np.array2string(cm)+"\n\n"+report)
print("test accuracy",float(np.mean(np.array(ys)==np.array(ps))))
print(cm)
print(report)
