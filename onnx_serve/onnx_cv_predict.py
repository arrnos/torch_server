import joblib
import onnxmltools
import onnxruntime as rt
import numpy as np
from sklearn import svm, datasets
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from datetime import datetime
import threading

# 创建和训练模型
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr').fit(X, y)

# 定义输入张量类型
initial_type = [('float_input', FloatTensorType([None, 2]))]

# 使用convert_sklearn函数将Scikit-learn模型转换为ONNX格式
onx = convert_sklearn(model, initial_types=initial_type)

# 将ONNX模型保存到本地文件中，并记录元数据
model_name = 'iris_svc'
version = '1.0'
model_path = f'{model_name}_{version}.pkl'
onnx_path = f'{model_name}_{version}.onnx'
joblib.dump(model, model_path)
onnxmltools.utils.save_model(onx, onnx_path)

# 加载模型和元数据
loaded_model = joblib.load(model_path)
print('Loaded Model:', loaded_model)

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


@app.route('/predict', methods=['POST'])
def make_prediction():
    data = request.get_json()
    prediction = predict(data['features'])
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    # 启动Flask应用程序，并配置多线程处理请求
    app.run(threaded=True, debug=True)

'''
如果您的Web应用程序需要支持并发请求，那么您需要确保您的代码是
线程安全的，并且可以处理多个并发请求。以下是一些可帮助您实现
这一目标的提示：


在启动应用程序时，使用threaded=True参数将Flask应用程序配置为使用多个线程处理请求，例如：app.run(threaded=True)。


如果您使用的是基于进程的服务器（如Gunicorn或uWSGI），则需要正确地调整工作进程数和线程数，以便最大化系统资源的利用率，并防止超过系统的限制。
这通常需要进行一些性能测试和基准测试，以找到最佳的设置。


对于模型推理，在每个请求中创建一个新的ONNX Runtime会话对象可能会影响性能，因为每个对象都需要占用一定的系统资源。为了避免这种情况，您可以使
用以下两种方法之一：


1. 使用线程本地存储TLS(Thread-Local Storage)来缓存已经创建的ONNX Runtime会话对象，以避免在每个请求中创建新的会话对象。这样可以提高性能，并减少内存消耗。

2. 使用基于进程的服务器，则可以使用进程池来缓存ONNX Runtime会话对象。这样，每个进程只需要创建一次会话对象，并在需要时重复使用该对象。这
种方法可以大大减少内存消耗，但也可能会对性能产生一些影响。

对于模型推理，您还可以考虑使用异步编程模型来处理并发请求。例如，可以使用asyncio库和aiohttp库来创建异步Web应用程序，并在每个请求中使用异步ONNX Runtime客户端来执行模型推理。

可以在上面的基础上添加并发控制。该代码使用Python的threading.local()方法创建了一个TLS(Thread-Local Storage)，以缓存ONNX Runtime会话对
象，并避免在每个请求中创建新的会话对象。同时，该代码还通过设置Flask应用程序的线程数来配置多线程处理请求。

在这个例子中，我们创建了一个名为tls的TLS(Thread-Local Storage)对象，以缓存ONNX Runtime会话对象。

这样可以避免每个请求都创建新的会话对象，从而提高性能和可扩展性。在predict()函数中，
我们首先检查TLS对象是否包含ONNX Runtime会话对象。如果没有，则创建一个新的会话对象，
并将其绑定到TLS对象上。然后，我们使用该会话对象执行模型推理。

最后，我们在启动Flask应用程序时，使用threaded=True参数来配置多线程处理请求。
这样可以利用系统的多核CPU资源，并提高应用程序的并发性能。

需要注意的是，虽然TLS可以避免在每个请求中创建新的会话对象，但它仍然可能导致竞态条件或死锁等问题。
'''