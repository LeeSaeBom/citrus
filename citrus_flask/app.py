from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
import base64
from efficientnet_pytorch import EfficientNet
import torch, torchvision
from torchvision import transforms
from PIL import Image

app = Flask(__name__)
app.secret_key = 'random string'

path = "model/best_efficientnet-95.pth"
device = torch.device('cpu')

num_classes = 6
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
model.load_state_dict(torch.load("model/best_efficientnet-95.pth", map_location=device))

model = model.to(device)
model.eval()

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


@app.route('/')
def index():
    return render_template('index.html')


# 업로드 HTML 렌더링
@app.route('/upload')
def render_file():
    return render_template('upload.html')


# 파일 업로드 처리
@app.route('/Uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/' + secure_filename(f.filename))
        image = Image.open(f).convert('RGB')

        image = transforms_test(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
        print(preds)
        if preds == 0:
            pred_class = 'citrus fruit xcc'
        elif preds == 1:
            pred_class = 'citrus fruit normal'
        elif preds == 2:
            pred_class = 'citrus leaf xcc'
        elif preds == 3:
            pred_class = 'citrus leaf red mite'
        elif preds == 4:
            pred_class = 'citrus leaf normal'
        else:
            pred_class = 'citrus leaf aphid'

        print('예측값은' + pred_class + '입니다')

        return render_template('upload.html', data=pred_class, img=f.filename)
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True,port = 8080)
