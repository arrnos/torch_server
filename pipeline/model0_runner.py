# 导入相关库
import torch
from torch import nn
from models.modelv0 import FashionMNISTModelV0
from datasets.fasion import get_train_data_loader, get_test_data_loader
from pipeline.base_runner import Runner
from utils.helper_functions import accuracy_fn

# 配置
batch_size = 32
device = 'cpu'
epochs = 1


def get_runner():
    # 构建模型
    model = FashionMNISTModelV0(input_shape=784,  # one for every pixel (28x28)
                                hidden_units=10,  # how many units in the hiden layer
                                output_shape=10  # one for every class
                                )

    # 损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()  # this is also called "criterion"/"cost function" in some places
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    runner = Runner(model, get_train_data_loader,
                    get_test_data_loader, epochs, batch_size, device,
                    optimizer, loss_fn, accuracy_fn
                    )
    return runner


if __name__ == '__main__':
    runner = get_runner()
    runner.train()
    it = iter(runner.test_data_loader)
    for i in range(20):
        inputs = next(it)[0]

        y_pred1 = runner.predict(inputs).detach().numpy()
        y1 = y_pred1.argmax(axis=1)[:10]
        print(y1)

        # 1.全部导出
        # runner.save("data/model/model2.pt")
        # runner.load("data/model/model2.pt")

        # 2.仅导出变量
        # runner.save("data/model/model0_state.pt", only_state=True)
        # runner.load("data/model/model0_state.pt",
        #             model=FashionMNISTModelV0(input_shape=784,  # one for every pixel (28x28)
        #                                       hidden_units=10,  # how many units in the hiden layer
        #                                       output_shape=10  # one for every class
        #                                       ))

        # 3.导出onnx
        runner.export_to_onnx("data/model/model0.onnx", input_shape=(1, 28, 28))
        y_pred2 = runner.predict_from_onnx("data/model/model0.onnx", inputs.numpy())
        y2 = y_pred2.argmax(axis=1)[:10]
        print(y2)
        # print(abs(y_pred1-y_pred2))
        print(abs((torch.Tensor(y_pred1) - torch.Tensor(y_pred2))).sum())
        print(accuracy_fn(torch.Tensor(y1), torch.Tensor(y2)))
