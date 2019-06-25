# Stock Market Prediction by Recurrent Neural Network

## Alpha Vantage

Create an Alpha Vantage API key [here](https://www.alphavantage.co/) and store at `secrets/alpha-vantage.key`.

## Install Kubeflow

```
kfctl init kfsmp
cd kfsmp
kfctl generate all -V
kfctl apply all -V
```