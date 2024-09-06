import os
import io
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import base64

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can limit to a specific origin instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define class mapping
class_mapping = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7',
    8: '8', 9: '9', 10: 'Da', 11: 'Dha', 12: 'Gah', 13: 'H', 14: 'Ii',
    15: 'Jah', 16: 'Lah', 17: 'M', 18: 'Na', 19: 'Sa', 20: 'Ta', 21: 'Tha',
    22: 'Uu', 23: 'a', 24: 'aa', 25: 'ai', 26: 'au', 27: 'ba', 28: 'bha',
    29: 'cha', 30: 'chha', 31: 'dah', 32: 'dhah', 33: 'e', 34: 'ga',
    35: 'gha', 36: 'ha', 37: 'i', 38: 'jJa', 39: 'ja', 40: 'jha', 41: 'ka',
    42: 'kha', 43: 'la', 44: 'ma', 45: 'nah', 46: 'o', 47: 'pah', 48: 'pha',
    49: 'ra', 50: 'sah', 51: 'tah', 52: 'thah', 53: 'tra', 54: 'u', 55: 'va',
    56: 'ya', 57: 'za'
}


# Initialize and load the model
def initialize_model(num_classes=58):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialize_model().to(device)

# Load model weights
model.load_state_dict(torch.load('resnet18_devanagari.pth', map_location=device, weights_only=True))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define prediction function
def predict(image: Image.Image) -> str:
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        _, predicted = output.max(1)
        return class_mapping.get(predicted.item(), "Unknown")


# Homepage with JavaScript
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Devanagari Character Recognition</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                }
                .container {
                    width: 100%;
                    max-width: 600px;
                    padding: 20px;
                    background-color: #fff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    text-align: center;
                }
                h1 {
                    color: #333;
                }
                form {
                    margin: 20px 0;
                }
                input[type="file"] {
                    margin-bottom: 10px;
                }
                input[type="submit"] {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 4px;
                    background-color: #007bff;
                    color: #fff;
                    cursor: pointer;
                }
                input[type="submit"]:hover {
                    background-color: #0056b3;
                }
                .result {
                    margin-top: 20px;
                }
                .result p {
                    font-size: 18px;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Devanagari Character Recognition</h1>
                <form id="uploadForm">
                    <input type="file" id="fileInput" name="file" accept="image/*" required>
                    <input type="submit" value="Predict">
                </form>
                <button onclick="location.href='/draw'">Free Draw</button>
                <div id="result" class="result"></div>
            </div>
            <script>
                document.getElementById('uploadForm').addEventListener('submit', async function(event) {
                    event.preventDefault();
                    let formData = new FormData();
                    formData.append('file', document.getElementById('fileInput').files[0]);

                    let response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    let result = await response.json();
                    document.getElementById('result').innerHTML = '<p>Predicted Class: ' + result.prediction + '</p>';
                });
            </script>
        </body>
    </html>
    """


# Canvas Drawing Page
@app.get("/draw", response_class=HTMLResponse)
async def draw():
    return """
    <html>
        <head>
            <title>Draw a Character</title>
            <style>
                canvas {
                    border: 1px solid #000;
                }
            </style>
        </head>
        <body>
            <h1>Draw a Character</h1>
            <label for="canvasWidth">Width:</label>
            <input type="number" id="canvasWidth" value="400" min="100" max="800">
            <label for="canvasHeight">Height:</label>
            <input type="number" id="canvasHeight" value="400" min="100" max="800">
            <button onclick="updateCanvasSize()">Update Canvas Size</button>
            <br><br>
            <canvas id="drawCanvas" width="400" height="400"></canvas><br><br>
            <button onclick="saveDrawing()">Save Drawing</button>
            <button onclick="submitDrawing()">Submit Drawing</button>
            <button onclick="viewHistory()">View History</button>
            <button onclick="clearHistory()">Clear History</button>
            <script>
                var canvas = document.getElementById('drawCanvas');
                var ctx = canvas.getContext('2d');
                var drawing = false;

                canvas.addEventListener('mousedown', function() {
                    drawing = true;
                });
                canvas.addEventListener('mouseup', function() {
                    drawing = false;
                    ctx.beginPath();
                });
                canvas.addEventListener('mousemove', draw);

                function draw(event) {
                    if (!drawing) return;
                    ctx.lineWidth = 5;
                    ctx.lineCap = 'round';
                    ctx.strokeStyle = 'black';
                    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
                }

                function updateCanvasSize() {
                    var width = document.getElementById('canvasWidth').value;
                    var height = document.getElementById('canvasHeight').value;
                    canvas.width = width;
                    canvas.height = height;
                }

                function saveDrawing() {
                    var dataURL = canvas.toDataURL('image/png');

                    var whiteCanvas = document.createElement('canvas');
                    whiteCanvas.width = canvas.width;
                    whiteCanvas.height = canvas.height;
                    var whiteCtx = whiteCanvas.getContext('2d');

                    whiteCtx.fillStyle = '#ffffff';
                    whiteCtx.fillRect(0, 0, whiteCanvas.width, whiteCanvas.height);

                    whiteCtx.drawImage(canvas, 0, 0);

                    var dataURLWithBackground = whiteCanvas.toDataURL('image/png');
                    var link = document.createElement('a');
                    link.href = dataURLWithBackground;
                    link.download = 'drawing.png';
                    link.click();
                }

                function submitDrawing() {
                    var whiteCanvas = document.createElement('canvas');
                    whiteCanvas.width = canvas.width;
                    whiteCanvas.height = canvas.height;
                    var whiteCtx = whiteCanvas.getContext('2d');

                    whiteCtx.fillStyle = '#ffffff';
                    whiteCtx.fillRect(0, 0, whiteCanvas.width, whiteCanvas.height);

                    whiteCtx.drawImage(canvas, 0, 0);

                    var dataURL = whiteCanvas.toDataURL('image/png');
                    var formData = new FormData();
                    formData.append('file', dataURLtoBlob(dataURL), 'drawing_with_white_background.png');

                    fetch('/submit_drawing/', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert('Prediction: ' + data.prediction);
                    });
                }

                function dataURLtoBlob(dataURL) {
                    var byteString = atob(dataURL.split(',')[1]);
                    var mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
                    var ab = new ArrayBuffer(byteString.length);
                    var ia = new Uint8Array(ab);
                    for (var i = 0; i < byteString.length; i++) {
                        ia[i] = byteString.charCodeAt(i);
                    }
                    return new Blob([ab], { type: mimeString });
                }

                function viewHistory() {
                    window.open('/history', '_blank');
                }
                  function clearHistory() {
                    fetch('/clear_history/', {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        // Optionally, update the history view if visible
                        document.getElementById('result').innerHTML = '';  // Clear any displayed history
                    });
                }
            </script>
        </body>
    </html>
    """


drawing_history = []  # In-memory storage for drawing history

@app.get("/history", response_class=HTMLResponse)
async def get_history():
    history_html = "<html><head><title>Drawing History</title></head><body><h1>Drawing History</h1><ul>"
    for item in drawing_history:
        history_html += f"<li><img src='{item}' width='100' height='100' /></li>"
    history_html += "</ul></body></html>"
    return history_html

@app.post("/submit_drawing/")
async def submit_drawing(file: UploadFile = File(...)):
    try:
        file_location = f"temp_images/{file.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as f:
            f.write(await file.read())

        image = Image.open(file_location).convert('L')
        image_np = np.array(image)
        _, thresh = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            raise HTTPException(status_code=400, detail="No character found in the image.")

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        cropped_image = image_np[y:y+h, x:x+w]
        cropped_pil_image = Image.fromarray(cropped_image)
        cropped_pil_image = cropped_pil_image.resize((32, 32))
        image_tensor = transform(cropped_pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = probabilities.max(1)
            predicted_class = class_mapping.get(predicted.item(), "Unknown")

        os.remove(file_location)

        # Add image to history
        dataURL = io.BytesIO()
        cropped_pil_image.save(dataURL, format='PNG')
        history_image_url = f"data:image/png;base64,{base64.b64encode(dataURL.getvalue()).decode()}"
        drawing_history.append(history_image_url)

        return {"prediction": predicted_class, "probabilities": probabilities.cpu().numpy().tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Prediction endpoint
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    prediction = predict(image)
    return {"prediction": prediction}


@app.post("/clear_history/")
async def clear_history():
    global drawing_history
    drawing_history = []  # Clear the history
    return {"message": "Drawing history cleared."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
