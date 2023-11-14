# 导入相关库
import numpy
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm
import threading
import onnxruntime as rt


class Runner(object):
    def __init__(self, model: torch.nn.Module, get_train_data_loader,
                 get_test_data_loader, epochs, batch_size, device,
                 optimizer, loss_fn, eval_fn, seed=42):

        torch.manual_seed(seed)
        self.model = model.to(device)
        self.get_train_data_loader = get_train_data_loader
        self.get_test_data_loader = get_test_data_loader
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = loss_fn.to(device)
        self.eval_fn = eval_fn
        self.optimizer = optimizer

        self.train_data_loader = None
        self.test_data_loader = None

        # 支持onnx多线程预测
        self.tls = threading.local()

    def train(self):
        # Set the seed and start the timer
        train_time_start_on_cpu = timer()
        self.model = self.model.to(self.device)

        # 构建data_loader
        self.train_data_loader = self.get_train_data_loader(self.batch_size)
        self.test_data_loader = self.get_test_data_loader(self.batch_size)

        # Create training and testing loop
        for epoch in tqdm(range(self.epochs)):
            print(f"Epoch: {epoch}\n-------")
            # Training
            train_loss = 0
            # Add a loop to loop through training batches
            for batch, (X, y) in enumerate(self.train_data_loader):
                self.model.train()
                X = X.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                train_loss += loss  # accumulatively add up the loss per epoch

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Print out how many samples have been seen
                if batch % 400 == 0:
                    print(f"Looked at {batch * len(X)}/{len(self.train_data_loader)} samples")

            # Divide total train loss by length of train data_loader (average loss per batch per epoch)
            train_loss /= len(self.train_data_loader)

            ### Testing
            # Setup variables for accumulatively adding up loss and accuracy
            test_loss, test_acc = 0, 0
            self.model.eval()
            with torch.inference_mode():
                for X, y in self.test_data_loader:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    # 1. Forward pass
                    test_pred = self.model(X)

                    # 2. Calculate loss (accumatively)
                    test_loss += self.loss_fn(test_pred, y)  # accumulatively add up the loss per epoch

                    # 3. Calculate accuracy (preds need to be same as y_true)
                    test_acc += self.eval_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

                test_loss /= len(self.test_data_loader)
                test_acc /= len(self.test_data_loader)
            print(f"\nTrain loss: {train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")

        train_time_end_on_cpu = timer()
        total_time = train_time_end_on_cpu - train_time_start_on_cpu
        print(f"Train time on cup: {total_time:.3f} seconds")

    def predict(self, inputs):
        predict = self.model(inputs)
        return predict

    def eval(self):
        """Returns a dictionary containing the results of model predicting on data_loader.

        Args:
            model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
            data_loader (torch.utils.data.data_loader): The target dataset to predict on.
            self.loss_fn (torch.nn.Module): The loss function of model.
            accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

        Returns:
            (dict): Results of model making predictions on data_loader.
        """
        loss, acc = 0, 0
        self.model.eval()
        # 加载验证数据
        if self.test_data_loader is None:
            self.test_data_loader = self.get_test_data_loader(self.batch_size)

        with torch.inference_mode():
            for X, y in self.test_data_loader:
                # Make predictions with the model
                y_pred = self.model(X)

                # Accumulate the loss and accuracy values per batch
                loss += self.loss_fn(y_pred, y)
                acc += self.eval_fn(y_true=y,
                                    y_pred=y_pred.argmax(
                                        dim=1))  # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)

            # Scale loss and acc to find the average loss/acc per batch
            loss /= len(self.test_data_loader)
            acc /= len(self.test_data_loader)

        eval_result = {"model_name": self.model.__class__.__name__,  # only works when model was created with a class
                       "model_loss": loss,
                       "model_acc": acc}
        print(eval_result)
        return eval_result

    def load(self, path, model=None):
        # 加载模型结构和状态参数
        if model is None:
            print("load whole model..")
            self.model = torch.load(path)
        # 新建模型结构、仅加载状态参数
        else:
            print("only load state dict..")
            self.model = model
            self.model.load_state_dict(torch.load(path))
        return self.model

    def save(self, path, only_state=False):
        self.model.to("cpu")
        # 仅保持状态字典
        if only_state:
            print("only save state dict..")
            torch.save(self.model.state_dict(), path)
        # 状态和结构都保存
        else:
            print('save whole model..')
            torch.save(self.model, path)

    def export_to_onnx(self, path, input_shape):
        input_args = torch.randn(1, *input_shape)
        input_names = ["input"]
        output_names = ["output"]

        torch.onnx.export(
            self.model,
            input_args,
            path,
            verbose=False,
            opset_version=11,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                "input": {0: "batch_size"},
                # list value: automatic names
                "output": [0],
            }

        )

    # 加载ONNX模型并执行推理
    def predict_from_onnx(self, path, data: numpy.ndarray):
        # 从TLS中获取或创建ONNX Runtime会话对象
        if not hasattr(self.tls, 'sess'):
            self.tls.sess = rt.InferenceSession(path)
        sess = self.tls.sess

        # 获取模型输入和输出名称
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

        # 执行模型推理
        return sess.run([output_name], {input_name: data})[0]
