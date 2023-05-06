# TUDelft_Interview_Unsupervised_Pretraining_with_PSE

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

## Results

| model | Ensemble Test Accuracy <br /> (top1, top2, top4) | Test 1 <br /> (acc, loss, num epochs) | Test 2 | Test 3 | Test 4| Test 5  |
|-------|-------------------------------------------|--------------------|--------|--------|-------|----------|
|CNN MSE | **46.76**, 65.73, 85.95 | **44.79**, 1.5380, 395  | **41.87**, 1.6121, 231  | **43.32**, 1.5710, 285  | **43.12**, 1.5808, 300 | **43.30**, 1.5674, 305 |
|CNN PSE | **47.98**, 66.72, 86.62 | **43.32**, 1.5840, 242  | **46.09**, 1.5207, 418  | **45.99**, 1.5237, 403  | **45.37**, 1.5559, 284 | **44.42**, 1.5351, 400 |
|RESNET MSE | **50.80**, 69.38, 87.25 | **50.11**, 1.4032, 276  | **47.37**, 1.5207, 148  | **50.11**, 1.4019, 274  | **49.78**, 1.4067, 256 | **46.37**, 1.5068, 134 |
|RESNET PSE | **53.25**, 71.91, 88.67 | **52.74**, 1.3247, 286  | **50.67**, 1.3771, 164  | **49.13**, 1.4242, 135  | **50.82**, 1.3729, 173 | **51.84**, 1.3540, 228 |
