import pandas as pd

N_original = pd.read_csv("N_original.csv")


# The following chunk of code was suggested by ChatGPT
#
# Prompt:
# "I have a Pandas Dataframe with a column with strings, but I want that columns data points to have strings of at most 300 words.
# How do I alter the dataframe such that the rows with strings of over 300 words are copied, except the string itself,
# which is divided over the copied rows, such that it has 300 words per row, at most?"
#
# (Partial) output:
# "Solution:
MAX_WORDS = 300

def split_text(text, max_words=MAX_WORDS):
    words = text.split()
    return [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]
# df["text"] = df["text"].apply(split_text)
#
# df = df.explode("text", ignore_index=True)"
N_original["post"] = N_original["post"].apply(split_text)

N_original = N_original.explode("post", ignore_index=True)

E_original = pd.read_csv("E_original.csv")
E_original["post"] = E_original["post"].apply(split_text)

E_original = E_original.explode("post", ignore_index=True)

E_cleaned = pd.read_csv("E_cleaned.csv")
E_cleaned["post"] = E_cleaned["post"].apply(split_text)

E_cleaned = E_cleaned.explode("post", ignore_index=True)
# End of chunk suggested by ChatGPT


# Model setup, as suggested by the author, Matous Volf
# https://huggingface.co/matous-volf/political-leaning-politics?
from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="matous-volf/political-leaning-politics",
    tokenizer="launch/POLITICS",
    device=0    # -1 for CPU, 0 for GPU. Results for this assignment were ran on an NVIDIA RTX 5070 GPU, in batches of 256
)


# The following code chunk was suggested by ChatGPT
# After running into many different errors, various prompts were used, asking about the errors and safe ways to store results in case something would go wrong again
# (Partial) output:
# "Safe way #1 (BEST): assign by index slice inside the loop
#
# This is exactly what you want if you’re worried about losing progress.
#
# df["label"] = None
# df["score"] = None
#
# batch_size = 32
#
# for i in range(0, len(df), batch_size):
#     idx = df.index[i:i+batch_size]
#     batch = df.loc[idx, "text"].tolist()
#
#     out = clf(batch, batch_size=batch_size, truncation=True)
#
#     df.loc[idx, "label"] = [o["label"] for o in out]
#     df.loc[idx, "score"] = [o["score"] for o in out]"
E_cleaned["label"] = None
E_cleaned["score"] = None

batch_size = 256

for i in range(0, len(E_cleaned), batch_size):
    print(i)
    idx = E_cleaned.index[i:i+batch_size]
    batch = E_cleaned.loc[idx, "post"].tolist()

    out = pipe(batch, batch_size=batch_size, truncation=True)

    E_cleaned.loc[idx, "label"] = [o["label"] for o in out]
    E_cleaned.loc[idx, "score"] = [o["score"] for o in out]

# E_cleaned.to_csv("results.csv") # Un-comment if you wish to store the DataFrame as a file

E_original["label"] = None
E_original["score"] = None

batch_size = 256

for i in range(0, len(E_original), batch_size):
    print(i)
    idx = E_original.index[i:i+batch_size]
    batch = E_original.loc[idx, "post"].tolist()

    out = pipe(batch, batch_size=batch_size, truncation=True)

    E_original.loc[idx, "label"] = [o["label"] for o in out]
    E_original.loc[idx, "score"] = [o["score"] for o in out]

# E_original.to_csv("results2.csv") # Un-comment if you wish to store the DataFrame as a file

N_original["label"] = None
N_original["score"] = None

batch_size = 256

for i in range(0, len(N_original), batch_size):
    print(i)
    idx = N_original.index[i:i+batch_size]
    batch = N_original.loc[idx, "post"].tolist()

    out = pipe(batch, batch_size=batch_size, truncation=True)

    N_original.loc[idx, "label"] = [o["label"] for o in out]
    N_original.loc[idx, "score"] = [o["score"] for o in out]

# N_original.to_csv("results3.csv") # Un-comment if you wish to store the DataFrame as a file
# This code chunk runs the model on all data sets and stores the predicted labels and corresponding scores in the respective DataFrames
# At the end, the resultsing DataFrame is stored in a csv file
# It stores the prediction per batch, to prevent loss of results
# This was added after running into many blue screens, CPU temperatures of > 100 degrees Celsius, after the model was running for hours
# The final results were created by running the model on GPU instead of CPU, with the code above.
# End of chunk suggested by ChatGPT