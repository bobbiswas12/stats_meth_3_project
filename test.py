
## logic for predicting and testing new values

import numpy as np
import math
from model import X, Y, B, columns, sigmoidf, n, m

X_new_dict = {}
X_new = []

for col in columns:
    print("Enter", col, ":")
    X_new_dict[col] = float(input())
    X_new.append(X_new_dict[col])

X_new = np.array(X_new, dtype=float).reshape(1, -1)

X_new_1 = np.ones((1, 1))
X_new = np.concatenate((X_new_1, X_new), axis=1)

def predict(X_new):
    p = sigmoidf((X_new @ B).item())

    print("Probability of stroke:", p)

    if p >= 0.5:
        print("Stroke: 1")
    else:
        print("Stroke: 0")

predict(X_new)

