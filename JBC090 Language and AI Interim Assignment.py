import pandas as pd
#data = pd.read_csv("reddit_pol.csv", na_values={"[removed]", "[deleted]"})
#clean_data = data[['subreddit', 'body']].dropna().reset_index(drop=True)
data = pd.read_csv("political_leaning.csv")
#print(data)

#clean_data = data.dropna().reset_index(drop=True)
#print(clean_data)

#print(clean_data.loc[309999]["body"])

count_left = data.post.str.count("left")
#print(count_left)

#print(data.post[0])

count_center = data.post.str.count("center")
count_right = data.post.str.count("right")
count_mentions = count_left + count_center + count_right
#print(count_mentions)

#print(count_center)
#print(count_right)

#print(type(count_mentions))

#more_data = pd.concat([data, count_mentions], axis=1).rename(["author_ID", "post", "political_leaning", "mentions"], axis=1)

more_data = pd.concat([data, count_mentions], axis=1, keys=["data", "mentions"])
#print(more_data)

#E_original = more_data[more_data.mentions.post > 0].data.set_index("index").reset_index()

# E_original = more_data[more_data.mentions.post > 0].data.reset_index(inplace=True)

E_original = more_data[more_data.mentions.post > 0].data.reset_index(drop=True)
print(E_original)

# N_original = more_data[more_data.mentions.post == 0].data.set_index("index").reset_index()

# N_original = more_data[more_data.mentions.post == 0].data.reset_index(inplace=True)

N_original = more_data[more_data.mentions.post == 0].data.reset_index(drop=True)
print(N_original)

# print(type(E_original))

# E_original.to_csv("E_original.csv")
# N_original.to_csv("N_original.csv")