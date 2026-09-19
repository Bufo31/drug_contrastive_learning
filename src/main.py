from train import train
from evaluate import evaluate

epochs = 5
batch_size = 32
learning_rate = 0.001

mode = "evaluate"

if mode == "train":
    train(epochs,batch_size,learning_rate)

elif mode == "evaluate":
    evaluate(batch_size)
