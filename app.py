import os
import flask
from flask import jsonify,request,send_file
from gevent.pywsgi import WSGIServer
from unet import UNet
import torch
import logging
from PIL import Image
import numpy as np
from utils.data_loading import BasicDataset
from utils.utils import save_image_overlay
import io
import base64

def create_app(config_filename):
    app = flask.Flask(__name__)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cpu')
    try:
        model = UNet(n_channels=3, n_classes=2, bilinear=False)
        state_dict = torch.load('weights/final_model.pth', map_location=device)       
        model.load_state_dict(state_dict)
        logging.info("Model initiated")
    except:
        raise Exception("no trained model found")
        logging.info("Model initiation failed")

    @app.errorhandler(400)
    def value_error(e):
        return jsonify(error=str(e)), 400

    @app.route('/health', methods=['GET'])
    def health_check():
        status = {200: "Container running successfully"}
        return jsonify(status)

    @app.route('/segment', methods=['POST'])
    def segment():
        data = request.files['image']
        img = data.read()
        img = Image.open(io.BytesIO(img))
        img = torch.from_numpy(BasicDataset.preprocess(img,0.5, is_mask=False)).to(device)
        img = img.unsqueeze(0)
        seg = model(img.type(torch.FloatTensor))
        seg = torch.argmax(seg,1)
        seg = seg.squeeze().cpu().numpy()
        img = np.asarray(img.squeeze().cpu().numpy())
        overlay = save_image_overlay(img,seg,path=None,epoch=None)
        
        return send_file(
        io.BytesIO(overlay.tobytes()),
        download_name='segmentation.png',
        mimetype='image/png'
    )
    return app

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    http_server = WSGIServer(('0.0.0.0', port), create_app('production'))
    http_server.serve_forever()