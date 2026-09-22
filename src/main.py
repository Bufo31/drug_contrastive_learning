from train import train
from evaluate import evaluate

epochs = 10
batch_size = 256
learning_rate = 1e-3

mode = "2"   #   1：train   2：evaluate

if mode == "1":
    train(epochs,batch_size,learning_rate)

elif mode == "2":
    evaluate(batch_size)