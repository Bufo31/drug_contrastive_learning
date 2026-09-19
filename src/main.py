from train import train
from evaluate import evaluate

epochs = 20
batch_size = 256
learning_rate = 1e-3

mode = "train"

if mode == "train":
    train(epochs,batch_size,learning_rate)

elif mode == "evaluate":
    evaluate(batch_size)