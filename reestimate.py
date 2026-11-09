
import numpy as np
import pandas as pd

from model import (
    X, Y, columns,
    B_logit, B_probit,
    IRLS, PROBIT_IRLS
)


# false positive observations (i am considering both the models)
selected_observations = [163, 246, 183, 134]

selected_indices = np.array(selected_observations) - 1

# remove false positive observations 
X_reduced = np.delete(X, selected_indices, axis=0)
Y_reduced = np.delete(Y, selected_indices, axis=0)

print("Original sample size:", len(Y))
print("Reduced sample size:", len(Y_reduced))
print("Observations removed:", selected_observations)


## re-estimating the model again (both logit and probit)


B_logit_reduced = IRLS(X_reduced, Y_reduced)
B_probit_reduced = PROBIT_IRLS(X_reduced, Y_reduced)

# var names
if len(columns) + 1 == X.shape[1]:
    variable_names = ["Intercept"] + list(columns)
elif len(columns) == X.shape[1]:
    variable_names = list(columns)
else:
    variable_names = [
        f"Beta_{i}" for i in range(X.shape[1])
    ]

comparison = pd.DataFrame({
    "Variable": variable_names,

    "Logit Original": B_logit.flatten(),
    "Logit Reduced": B_logit_reduced.flatten(),

    "Probit Original": B_probit.flatten(),
    "Probit Reduced": B_probit_reduced.flatten()
})

# coeff. changes
comparison["Logit Change"] = (
    comparison["Logit Reduced"] -
    comparison["Logit Original"]
)

comparison["Probit Change"] = (
    comparison["Probit Reduced"] -
    comparison["Probit Original"]
)


print(comparison.to_string(
    index=False,
    float_format=lambda x: f"{x:.8f}"
))

comparison.to_csv(
    "coefficient_comparison.csv",
    index=False
)


logit_norm = np.linalg.norm(B_logit_reduced - B_logit)
l_norm = np.linalg.norm(B_logit)
p_norm = np.linalg.norm(B_probit)
probit_norm = np.linalg.norm(B_probit_reduced - B_probit)

print("||Beta_logit_reduced - Beta_logit_full|| =", logit_norm)
print("||Beta_probit_reduced - Beta_probit_full|| =", probit_norm)
print("||Beta_logit_full|| =",l_norm)
print("||Beta_probit_full|| =",p_norm)
