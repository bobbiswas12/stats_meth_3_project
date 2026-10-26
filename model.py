

## here comes the logic and implementation for the actual model


import numpy as np
import math
from loader import load_data

X,Y = load_data("datasets/healthcare-dataset-stroke-data.csv")

# the sigmoid function --->
def sigmoidf(x):

    return 1/(1 + np.exp(-x))
# the sigmoid function--->

# THE LOGISTIC REGRESSION MODEL PART -

def model(X,Y):

    n,m = X.shape
    B = np.zeros((m,1))

    l = 0
    i = 0
    
    for x_i in X:
        
        p_i = sigmoidf((x_i @ B).item())
        l += Y[i] * math.log(p_i) + (1 - Y[i]) * math.log(1 - p_i)
        i += 1

    print(l)

model(X,Y)
# print(y,y.shape)
# print(x,x.shape)
# print(b,b.shape)


