import numpy as np
import pandas as pd
from scipy.stats import norm
from model import X, Y, B_probit, B_logit, sigmoidf

# i am using 0.01 as the threshold, basically flag observations if their linear predictor (eta) < 0.01 and their outcome (Y) is 1
threshold = 0.01

X_with_intercept = X ## my bad, i had already included the intercept values in X before

Y = np.asarray(Y).reshape(-1)

## eta for both models
eta_probit = (X_with_intercept @ B_probit).reshape(-1)
eta_logit = (X_with_intercept @ B_logit).reshape(-1)

## fit probabilities for both models
p_probit = norm.cdf(eta_probit)
p_logit = sigmoidf(eta_logit)

## score weights for positive observations
score_weight_probit = (
    norm.pdf(eta_probit) / np.maximum(p_probit, 1e-300)
)

score_weight_logit = 1 - p_logit # only true for logit as D(1/1+e^-x)/Dx = e^-x/1+e^-x)

## all positive observations
positive_indices = np.where(Y == 1)[0]

results = pd.DataFrame({
    "Observation": positive_indices + 1,
    "Probit Probability": p_probit[positive_indices],
    "Logit Probability": p_logit[positive_indices],
    "Actual Outcome": Y[positive_indices],
    "Probit Score Weight": score_weight_probit[positive_indices],
    "Logit Score Weight": score_weight_logit[positive_indices]
})

# just flagging those outlier observations
results["Low Probit"] = results["Probit Probability"] < threshold
results["Low Logit"] = results["Logit Probability"] < threshold

unexpected_both = results[
    results["Low Probit"] & results["Low Logit"]
]

unexpected_either = results[
    results["Low Probit"] | results["Low Logit"]
]

results = results.sort_values(
    by=["Probit Probability", "Logit Probability"]
)

unexpected_both = unexpected_both.sort_values(
    by=["Probit Probability", "Logit Probability"]
)

unexpected_either = unexpected_either.sort_values(
    by=["Probit Probability", "Logit Probability"]
)

columns_to_display = [
    "Observation",
    "Probit Probability",
    "Logit Probability",
    "Actual Outcome",
    "Probit Score Weight",
    "Logit Score Weight",
    "Low Probit",
    "Low Logit"
]

print("no. of total positive observations:", len(positive_indices))

print("all positive observations:")
print(results[columns_to_display].to_string(
    index=False,
    float_format=lambda x: f"{x:.8f}"
))

print("unexpected positive observations under BOTH models:")
print(unexpected_both[columns_to_display].to_string(
    index=False,
    float_format=lambda x: f"{x:.8f}"
))

print("unexpected positive observations under EITHER model:")
print(unexpected_either[columns_to_display].to_string(
    index=False,
    float_format=lambda x: f"{x:.8f}"
))

print("number unexpected under both models:", len(unexpected_both))
print("number unexpected under either model:", len(unexpected_either))


selected_observations = [163, 246, 183, 134]

# Convert to Python indices (0-based indexing)
selected_indices = np.array(selected_observations) - 1

# Remove selected observations
X_reduced = np.delete(X, selected_indices, axis=0)
Y_reduced = np.delete(Y, selected_indices, axis=0)

print("Original sample size:", len(Y))
print("Reduced sample size:", len(Y_reduced))
print("Observations removed:", len(selected_indices))

print("i am flagging all the positive observations that have linear predictor value less than 0.01 under EITHER of the models")
