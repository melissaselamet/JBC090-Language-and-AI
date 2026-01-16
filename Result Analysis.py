import pandas as pd
import numpy as np

N_original = pd.read_csv("results3.csv")

N_original["label_num"] = N_original["label"].str.extract(r"(\d+)$").astype(int)    # Stores the predicted labels as integers

N_original["probabilities"] = (N_original["score"] - (1 / 3)) / (2 / 3) # Translates the score (between 1/3 and 1) to percentages to use as weights (1/3 = 0%, 1 = 100%)

# Calculates a weighted average of the predictions for each author, using the translated scores as weights, and stores it in a Pandas Groupby object
N_original["average_weights"] = N_original["probabilities"] * N_original["label_num"]   # 
prob_sums = N_original.groupby("auhtor_ID").agg('sum')[["probabilities", "average_weights"]]
prob_sums["prediction"] = prob_sums["average_weights"] / prob_sums["probabilities"]

# Saves the Groupby object
prob_sums.to_csv("clean_predictions3.csv")
prob_sums = pd.read_csv("clean_predictions3.csv")

prob_sums["prediction"] = prob_sums["prediction"].apply(lambda x: round(x)) # Rounds predictions to integers

# Translates the original labels from strings to corresponding integers
conditions = [
    N_original['political_leaning'] == "left",
    N_original['political_leaning'] == "center",
    N_original['political_leaning'] == "right"
]
choices = ["0", "1", "2"]
N_original['leaning_translated'] = np.select(conditions, choices, default='Unknown')
N_original["leaning_translated"] = pd.to_numeric(N_original["leaning_translated"])

# Ensures that the author IDs match in data types in both the original DataFrame and the Groupy object
prob_sums["auhtor_ID"] = prob_sums["auhtor_ID"].astype(pd.StringDtype())
N_original["auhtor_ID"] = N_original["auhtor_ID"].astype(pd.StringDtype())

# Merges the author IDs and their corresponding labels with the Groupby object
df_final1 = prob_sums.merge(N_original[["auhtor_ID", "leaning_translated"]], how='inner', on="auhtor_ID").drop_duplicates()

# Calculates and prints the accuracy of the model
accuracy = len(df_final1[df_final1["prediction"] == df_final1["leaning_translated"]]) / len(df_final1)
print(accuracy)

# Prints a confusion matrix
def confusion_matrix1(prediction, actual):
    return df_final1[df_final1["prediction"] == prediction]["leaning_translated"].value_counts().get(actual)
print("Prediction 0 vs Actual:", confusion_matrix1(0, 0), confusion_matrix1(0, 1), confusion_matrix1(0, 2))
print("Prediction 1 vs Actual:", confusion_matrix1(1, 0), confusion_matrix1(1, 1), confusion_matrix1(1, 2))
print("Prediction 2 vs Actual:", confusion_matrix1(2, 0), confusion_matrix1(2, 1), confusion_matrix1(2, 2))


# The remaining code is the exact same as the code above, but adapted to their respective data sets
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
E_original = pd.read_csv("results2.csv")

E_original["label_num"] = E_original["label"].str.extract(r"(\d+)$").astype(int)    # Stores the predicted labels as integers

E_original["probabilities"] = (E_original["score"] - (1 / 3)) / (2 / 3) # Translates the score (between 1/3 and 1) to percentages to use as weights (1/3 = 0%, 1 = 100%)

# Calculates a weighted average of the predictions for each author, using the translated scores as weights, and stores it in a Pandas Groupby object
E_original["average_weights"] = E_original["probabilities"] * E_original["label_num"]   # 
prob_sums = E_original.groupby("auhtor_ID").agg('sum')[["probabilities", "average_weights"]]
prob_sums["prediction"] = prob_sums["average_weights"] / prob_sums["probabilities"]

# Saves the Groupby object
prob_sums.to_csv("clean_predictions2.csv")
prob_sums = pd.read_csv("clean_predictions2.csv")

prob_sums["prediction"] = prob_sums["prediction"].apply(lambda x: round(x)) # Rounds predictions to integers

# Translates the original labels from strings to corresponding integers
conditions = [
    E_original['political_leaning'] == "left",
    E_original['political_leaning'] == "center",
    E_original['political_leaning'] == "right"
]
choices = ["0", "1", "2"]
E_original['leaning_translated'] = np.select(conditions, choices, default='Unknown')
E_original["leaning_translated"] = pd.to_numeric(E_original["leaning_translated"])

# Ensures that the author IDs match in data types in both the original DataFrame and the Groupy object
prob_sums["auhtor_ID"] = prob_sums["auhtor_ID"].astype(pd.StringDtype())
E_original["auhtor_ID"] = E_original["auhtor_ID"].astype(pd.StringDtype())

# Merges the author IDs and their corresponding labels with the Groupby object
df_final2 = prob_sums.merge(E_original[["auhtor_ID", "leaning_translated"]], how='inner', on="auhtor_ID").drop_duplicates()

# Calculates and prints the accuracy of the model
accuracy = len(df_final2[df_final2["prediction"] == df_final2["leaning_translated"]]) / len(df_final2)
print(accuracy)

# Prints a confusion matrix
def confusion_matrix2(prediction, actual):
    return df_final2[df_final2["prediction"] == prediction]["leaning_translated"].value_counts().get(actual)
print("Prediction 0 vs Actual:", confusion_matrix2(0, 0), confusion_matrix2(0, 1), confusion_matrix2(0, 2))
print("Prediction 1 vs Actual:", confusion_matrix2(1, 0), confusion_matrix2(1, 1), confusion_matrix2(1, 2))
print("Prediction 2 vs Actual:", confusion_matrix2(2, 0), confusion_matrix2(2, 1), confusion_matrix2(2, 2))


E_cleaned = pd.read_csv("results.csv")

E_cleaned["label_num"] = E_cleaned["label"].str.extract(r"(\d+)$").astype(int)    # Stores the predicted labels as integers

E_cleaned["probabilities"] = (E_cleaned["score"] - (1 / 3)) / (2 / 3) # Translates the score (between 1/3 and 1) to percentages to use as weights (1/3 = 0%, 1 = 100%)

# Calculates a weighted average of the predictions for each author, using the translated scores as weights, and stores it in a Pandas Groupby object
E_cleaned["average_weights"] = E_cleaned["probabilities"] * E_cleaned["label_num"]   # 
prob_sums = E_cleaned.groupby("auhtor_ID").agg('sum')[["probabilities", "average_weights"]]
prob_sums["prediction"] = prob_sums["average_weights"] / prob_sums["probabilities"]

# Saves the Groupby object
prob_sums.to_csv("clean_predictions.csv")
prob_sums = pd.read_csv("clean_predictions.csv")

prob_sums["prediction"] = prob_sums["prediction"].apply(lambda x: round(x)) # Rounds predictions to integers

# Translates the original labels from strings to corresponding integers
conditions = [
    E_cleaned['political_leaning'] == "left",
    E_cleaned['political_leaning'] == "center",
    E_cleaned['political_leaning'] == "right"
]
choices = ["0", "1", "2"]
E_cleaned['leaning_translated'] = np.select(conditions, choices, default='Unknown')
E_cleaned["leaning_translated"] = pd.to_numeric(E_cleaned["leaning_translated"])

# Ensures that the author IDs match in data types in both the original DataFrame and the Groupy object
prob_sums["auhtor_ID"] = prob_sums["auhtor_ID"].astype(pd.StringDtype())
E_cleaned["auhtor_ID"] = E_cleaned["auhtor_ID"].astype(pd.StringDtype())

# Merges the author IDs and their corresponding labels with the Groupby object
df_final3 = prob_sums.merge(E_cleaned[["auhtor_ID", "leaning_translated"]], how='inner', on="auhtor_ID").drop_duplicates()

# Calculates and prints the accuracy of the model
accuracy = len(df_final3[df_final3["prediction"] == df_final3["leaning_translated"]]) / len(df_final3)
print(accuracy)

# Prints a confusion matrix
def confusion_matrix3(prediction, actual):
    return df_final3[df_final3["prediction"] == prediction]["leaning_translated"].value_counts().get(actual)
print("Prediction 0 vs Actual:", confusion_matrix3(0, 0), confusion_matrix3(0, 1), confusion_matrix3(0, 2))
print("Prediction 1 vs Actual:", confusion_matrix3(1, 0), confusion_matrix3(1, 1), confusion_matrix3(1, 2))
print("Prediction 2 vs Actual:", confusion_matrix3(2, 0), confusion_matrix3(2, 1), confusion_matrix3(2, 2))
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------