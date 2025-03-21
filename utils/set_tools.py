import math
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm

# 1️⃣ Convert lists to sets
# df_set = df['name_tokens'].apply(frozenset)


def jaccard_distance(set1, set2):
    return 1 - len(set1 & set2) / len(set1 | set2)  # Jaccard Index Complement


def dice_distance(set1, set2):
    # Sørensen-Dice
    return 1 - (2 * len(set1 & set2) / (len(set1) + len(set2)))


def compute_distance_matrix(sets, distance_func):
    n = len(sets)
    matrix = np.zeros((n, n))  # Initialize an NxN matrix

    # Iterate over unique pairs
    for i, j in tqdm(combinations(range(n), 2), total=math.comb(n, 2)):
        dist = distance_func(sets[i], sets[j])
        matrix[i, j] = matrix[j, i] = dist  # Symmetric matrix

    return matrix

# pd.DataFrame(matrix, index=df.index, columns=df.index)
