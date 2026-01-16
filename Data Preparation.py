import pandas as pd

data = pd.read_csv("political_leaning.csv")
clean_data = data.dropna().reset_index(drop=True)   # Removes any empty rows if there is any

# Counts all occurrences of the words "left", "center", and "right" for each post, stored in Pandas Series
count_left = data.post.str.count("left")
count_center = data.post.str.count("center")
count_right = data.post.str.count("right")

count_mentions = count_left + count_center + count_right    # Adds up the counts of all words per post, stored in a single Pandas Series

more_data = pd.concat([data, count_mentions], axis=1, keys=["data", "mentions"])    # Concatenates the original dataframe with the counts Series

# Filters out all posts with 0 mentions of the labels and stores it (in a csv-file)
E_original = more_data[more_data.mentions.post > 0].data.reset_index(drop=True)
# E_original.to_csv("E_original.csv")   # Un-comment if you wish to store the DataFrame as a file

# Prints how many of the remaining posts are labeled left, center, and right
print(len(E_original[E_original.political_leaning == "left"]))
print(len(E_original[E_original.political_leaning == "center"]))
print(len(E_original[E_original.political_leaning == "right"]))


# Running the following chunk of code requires the file earlier created in this code file: "E_original.csv"
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# E_cleaned = pd.read_csv("E_original.csv") # Un-comment if the file is, or will be by the time the code reaches this line, present in your current working directory

# Replaces all mentions of the labels with the empty string, in other words, removes mentions of the label
E_cleaned["post"] = E_cleaned.post.str.replace("left", "", regex=False)
E_cleaned["post"] = E_cleaned.post.str.replace("center", "", regex=False)
E_cleaned["post"] = E_cleaned.post.str.replace("right", "", regex=False)
# E_cleaned.to_csv("E_cleaned.csv") # Un-comment if you wish to store the DataFrame as a file
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------


N_original = more_data[more_data.mentions.post == 0].data.reset_index(drop=True)    # Filters out all posts that mention the labels at least once

# Prints how many of the remaining posts are labeled left, center, and right
print(len(N_original[N_original.political_leaning == "left"]))
print(len(N_original[N_original.political_leaning == "center"]))
print(len(N_original[N_original.political_leaning == "right"]))
# N_original.to_csv("N_original.csv") # Un-comment if you wish to store the DataFrame as a file