import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import serial
from PIL import Image
import numpy as np

# ===== DEVICE =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# ===== CLASSES =====
class_names = ['Cat', 'Dog']

# ===== OPTIONAL: Serial Setup =====
# ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

# ===== DEFINE MODEL CLASS =====
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomResNet50, self).__init__()
        self.conv_base = models.resnet50(weights='IMAGENET1K_V1')
        for param in self.conv_base.parameters():
            param.requires_grad = False
        self.conv_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.conv_base.fc.in_features
        self.conv_base.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if len(x.shape) == 5:
            x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        return self.conv_base(x)

# ===== LOAD FULL MODEL =====
model = torch.load("KV-dog-cat.pth", map_location=device)
model = model.to(device)
model.eval()

# ===== TRANSFORMS =====
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== CAMERA STREAM =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not captured.")
        break

    # Convert frame to PIL RGB image
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Preprocess and send to GPU
    input_tensor = val_transforms(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_class = class_names[pred_idx]
        confidence = probs[0][pred_idx].item()

    # Draw result
    label = f"{predicted_class} ({confidence:.2f})"
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Jetson Cat/Dog Classifier", frame)

    # Optional: Send to serial
    # try:
    #     ser.write(f"{predicted_class}\n".encode())
    # except Exception as e:
    #     print("Serial write error:", e)

    # Exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== CLEANUP =====
cap.release()
cv2.destroyAllWindows()
# ser.close()
