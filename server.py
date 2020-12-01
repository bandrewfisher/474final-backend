from flask import Flask, request
from flask_cors import CORS, cross_origin
from PIL import Image
import io
import numpy as np
import base64
import json
import torch.nn as nn
import torch
import torchvision.utils
from torchvision.transforms import ToTensor, GaussianBlur
import pickle

app = Flask(__name__)
cors = CORS(app)


class CharacterRecognition(nn.Module):
    def __init__(self):
        super(CharacterRecognition, self).__init__()
        self.convPool = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 5, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 300),
            nn.LeakyReLU()
        )

    def forward(self, x):
        conv_out = self.convPool(x)
        return self.fc(conv_out.view(-1, 2048))


model = CharacterRecognition()
model.load_state_dict(torch.load(
    'model9.pickle', map_location=torch.device('cpu')))
model.eval()

char_classes = pickle.load(open('train_classes.pickle', 'rb'))


@app.route('/recognize', methods=['POST'])
@cross_origin()
def recognize():
    drawing_base64 = request.get_json()['drawing'].split(',')[1]
    drawing_decoded = base64.b64decode(drawing_base64)

    image = Image.open(io.BytesIO(drawing_decoded))
    image_torch = torch.tensor(
        np.array(image)[:128, :128, :3]).permute(2, 0, 1).float()

    torchvision.utils.save_image(image_torch, 'image.png')
    classes = model(image_torch.unsqueeze(0))
    guess_classes = torch.softmax(classes, 1).squeeze(
        0).topk(10).indices.tolist()

    return json.dumps([char_classes[c] for c in guess_classes])


app.run(port=5000, debug=True)
