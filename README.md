# Unsupervised Pretraining with PSE

## Install requirements and prepare data

```
pip install -r requirements.txt
python stl10_input.py
```

## Pretraining

example:

```
python pretrain.py --model_type CNN --model_name AUTOENCODER_PRETRAIN_CNN_MSE --loss_type MSE
python pretrain.py --model_type RESNET --model_name AUTOENCODER_PRETRAIN_RESNET_PSE6 --loss_type PSE --PSE_sigma 6
```

## Downstream

example:

```
nohup python downstream.py --model_type CNN --model_name AUTOENCODER_PRETRAIN_CNN_MSE > DOWNSTREAM_CNN_MSE.txt &
nohup python downstream.py --model_type RESNET --model_name AUTOENCODER_PRETRAIN_RESNET_PSE6 > DOWNSTREAM_RESNET_PSE6.txt &
```
