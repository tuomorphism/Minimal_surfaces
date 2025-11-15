import numpy as np

def levi_civita_symbol(i, j, k):
    denominator = abs(j - i) * abs(k - i) * abs(k - j)
    numerator = (j - i) * (k - i) * (k - j)

    if denominator == 0:
        return 0
    return numerator / denominator

def generate_symbols() -> list:
    # Brute force calculation for combinations which have \epsilon_{i, j, k} = 1
    total_permutations = []
    for i in range(3):
        permutations = []
        for j in range(3):
            for k in range(3):
                if levi_civita_symbol(i, j, k) == 1:
                    permutations.append([i, j, k])
        total_permutations.append(permutations)
    return total_permutations