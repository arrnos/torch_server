import onnxmltools
import onnxruntime as rt
import numpy as np
import threading

onnx_path = "data/model/model0.onnx"

# 创建TLS以缓存ONNX Runtime会话对象
tls = threading.local()


# 加载ONNX模型并执行推理
def predict(data):
    # 从TLS中获取或创建ONNX Runtime会话对象
    if not hasattr(tls, 'sess'):
        tls.sess = rt.InferenceSession(onnx_path)
    sess = tls.sess

    # 获取模型输入和输出名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # 执行模型推理
    data = np.array(data, dtype=np.float32)
    return sess.run([output_name], {input_name: data})[0]


# 在Web应用程序中使用模型进行推理
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/model0/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    prediction = predict(data['input'])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    # 启动Flask应用程序，并配置多线程处理请求
    app.run(threaded=True, debug=True)
