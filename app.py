# app.py
# Separate Flask web page for uploading images and predicting digits
# Your training code remains unchanged

from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Same CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Load model
model = CNN().to(device)

# Load saved weights
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image = Image.open(filepath)

            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)

            prediction = predicted.item()
            image_path = filepath

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)