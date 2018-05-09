# Wide-resnet-fashionMnist

## 1.  Wide-resnet

The wide residual network is a variety of residual network and This paper ["Wide Residual Networks" by Sergey Zagoruyko, Nikos Komodakis, arXiv:1605.07146](https://arxiv.org/abs/1605.07146)  first proposed the wide residual network. You can find the implementation of the wide resnet in [here](https://github.com/szagoruyko/wide-residual-networks). But this codes in this project referenced the other implementation, you can find in [here](https://github.com/JRC1995/Wide-Residual-Network)

## 2. Fashion Mnist

You can understand the project in [here](https://github.com/zalandoresearch/fashion-mnist)

## 3. How to run
You can use the wideres.yaml file to create the environment the program can run, but many packages in this environment is necessary for you. The main packages needed in the program are tensorflow, matplotlib, h5py, pillow and so on.
You can find how to process the dataset in dataProcessing.py, dataDemo2.py and dataDemo3.py 

You can download the dataset with .h5 file in [here](https://pan.baidu.com/s/1tIAACbyN-7C1FRXhcvUF0g).
	cifar-10 processed_data.h5
	fashion-mnist fashion_data.h5

1. train for cifar-10
```python
python train.py
```
2. train for fashion mnist
```python
python trainfashion.py
```


