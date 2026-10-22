

## here comes the logic and implementation for the actual model




import numpy as np
import math

## 'n' samples of 'p' variables -
## Y_i = X_i^T B + E_i

##X_i = np.array([]) 1-dimensional array with p + 1 entries (including 1's)
##Y_i = 1 or 0 (depends on the dataset)

#say n = 10, and p = 5

def sigmoidf(x):

    return 1/(1 + np.exp(-x))


n = 10
p = 5

y = np.array([1,1,1,0,1,0,0,0,1,0])
x_1 = np.ones((n,1))
x = np.random.rand(n,p)
x = np.concatenate((x_1,x), axis = 1)
b = np.ones(((p + 1),1))



# THE LOGISTIC REGRESSION MODEL PART -

def logreg()

l = 0

for i in range(n):

    x_i = x[i, :]
    y_i = y[i]

    ## value of p_i = 1/(1 + e^-(x_i @ b)
    
    p_i = sigmoidf((x_i @ b).item())
    
    l += y_i * math.log(p_i) + (1 - y_i) * math.log(1 - p_i)



# IRLS for maximizing b -




print(l)

# print(y,y.shape)
# print(x,x.shape)
# print(b,b.shape)


