from nn import Perceptronmulticapa
import numpy as np

X, Y=[],[]
file = open("dataset_ejemplo_40_3_16.csv")
file.readline()
for line in file:
    line=line.strip().split(";")
    x=list(map(float,[line[1],line[2],line[3]]))
    y=[1,0] if line[0]=="R" else [0,1]
    X.append(x)
    Y.append(y)
file.close()
X=np.asarray(X)
Y=np.asarray(Y)
mlp = Perceptronmulticapa(hidden=32)
mlp.train(X,Y,epochs=128)