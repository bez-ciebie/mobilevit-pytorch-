![image](https://github.com/bez-ciebie/mobilevit-pytorch-train-infer/assets/47070146/ff3fb0da-eb93-4797-bdac-bf78d72d678d)# mobilevit-pytorch
根据网络结构自己修改的训练和推理的代码
训练+推理
权重是自己在miniimagenet上训练的，建议自己训练
这里我主要是要验证在边缘设备上可以正常推理，对模型的精度没有要求


对于需要进行算子融合操作的使用者，可能需要有以下改动
![image](https://github.com/bez-ciebie/mobilevit-pytorch-train-infer/assets/47070146/99d8988c-b1fc-4b07-ae87-ba0165a22fdf)

由于算子融合无法使用.shape获得参数，所以使用numel函数先查看cpu上的维度，在其他机器上注释掉维度获取，自己加上三次输入的维度的判断，算子融合就可以正确运行。


