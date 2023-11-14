from flask import Flask, request, jsonify
from pipeline.model0_runner import get_runner
from models.modelv0 import FashionMNISTModelV0
import time
import io
from PIL import Image
from flask import Flask, request, jsonify
import torchvision
import torch
import json
import time
from utils.util1 import *

app = Flask(__name__)

# 1.初始化模型
runner = get_runner()
model = runner.load("../data/model/model0_state.pt",
                    model=FashionMNISTModelV0(input_shape=784,  # one for every pixel (28x28)
                                              hidden_units=10,  # how many units in the hiden layer
                                              output_shape=10  # one for every class
                                              ))
print(model)
runner.eval()

# 2.图像转化transformer
transforms = torchvision.transforms.Compose([  # 归一化，Tensor处理
    torchvision.transforms.ToTensor()
])


# 预测入口
@app.route('/ship_detect/predict', methods=['POST'])
def post_api():
    data = request.get_json()
    print(data)
    res = {
        "msg": "",
        "image": ""
    }

    if request.files.get("image"):
        # print("world")
        now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
        # read the image in PIL format

        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image.save(now + '.jpg')

        # preprocess the image and prepare it for classification
        W, H, scale, img = narrow_image(image)  # 传入缩放函数中，得到W,H,缩放比例，缩放后的416*416的图片
        img_data = transforms(img).unsqueeze(0)

        # box = detector(img_data, 0.25, anchors_cfg.ANCHORS_GROUP)  # 将图片传入侦测
        # box = enlarge_box(W, H, scale, box)  # 得到的box是基于416图片，需反算回到原图
        res = json_text(box, W, H)  # 将box整理成json格式

    return res


if __name__ == '__main__':
    app.run(debug=True)
