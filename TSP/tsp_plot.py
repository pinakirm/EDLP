import matplotlib.pyplot as plt
import numpy as np
# Data for plotting
"""
metrics = ['2','4', '5', '8', '10']
samplers = ['GWG', 'DMALA', 'ACS', 'PT','HiSS']
values = [
    [1,1, 1, 1, 1] ,  # GWG
    [1,1, 1, 1, 1] ,  # DMALA
    [2,5, 6, 7, 6],  # ACS
    [1,1, 1, 1, 1] ,  # PT
    [2,7, 30, 6, 7]     #HiSS
]

# Transpose data for plotting
values_transposed = list(zip(*values))

# Create the plot
plt.figure(figsize=(8, 6))
for i, sampler in enumerate(samplers):
    plt.plot(metrics, [row[i] for row in values_transposed], marker='o', label=sampler)

plt.title('Comparison of Number of Solutions Across Samplers')
plt.xlabel('Number of Cities')
plt.ylabel('Number of Solutions')
plt.legend(title='Samplers')
plt.grid(True)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Data for the bar chart
categories = ['2','4', '5', '8', '10']
groups = samplers
means = [
    [24.7386, 102.5844, 122.6775, 206.7094, 310.5709],  # Kent
    [24.7386, 102.5844, 122.6775, 206.7094, 310.5709] ,  # Lincoln
    [24.7386,103.7495, 125.5322, 208.1433, 301.4464],  # Mersey
    [24.7386,102.5844, 122.6775, 206.7094, 310.5709] ,  # York
    [24.7386,108.7307, 134.7343, 193.2357, 295.4187]
]
std_devs = [
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # GWG
    [np.nan, np.nan, np.nan, np.nan, np.nan],  # DMALA
    [ 0.000,2.6052, 16.1875, 17.9041, 32.8387],  # ACS
    [np.nan, np.nan, np.nan, np.nan, np.nan], #  PT+DMALA
    [ 0.000,  7.0188, 16.2104, 9.2292, 20.3934]#  HiSS
]

# Bar chart parameters
x = np.arange(len(categories))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))

# Plot each group
for i, (group, mean, std) in enumerate(zip(groups, means, std_devs)):
    ax.bar(x + i * width, mean, width, label=group, yerr=std, capsize=5)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Number of Cities')
ax.set_ylabel('Average Cost')
ax.set_title('Cost Analysis for Sampled Solutions')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(categories)
ax.legend(title='Samplers')

plt.tight_layout()
plt.show()
"""


def jaccard_similarity(path1, path2):
    """Returns the Jaccard similarity of directed edges between two tours."""
    cycle1 = path1 + [path1[0]]
    cycle2 = path2 + [path2[0]]
    edges1 = set(zip(cycle1[:-1], cycle1[1:]))
    edges2 = set(zip(cycle2[:-1], cycle2[1:]))
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)
    return len(intersection) / len(union) if union else 0.0

# Example usage:


def pairwise_mismatch_count(path1, path2):
    path1=path1+[path1[0]]
    path2 = path2 + [path2[0]]

    distance = 0
    for i in range(len(path1)):
        for j in range(i + 1, len(path1)):
            # Check if the pairwise order is different in the two paths
            if (path1[i], path1[j]) != (path2[i], path2[j]):
                distance += 1
    return distance


SD = []
SD1=[]
chain_paths=[[5, 11, 6, 2, 0, 4, 3, 8, 10, 13, 12, 1, 9, 7], [11, 6, 4, 13, 7, 1, 2, 5, 12, 9, 10, 8, 0, 3]]



from itertools import combinations

for path, path1 in combinations(chain_paths, 2):  # Generates unique pairs (path, path1)
    SD.append(pairwise_mismatch_count(path, path1))
    SD1.append(jaccard_similarity(path, path1))



for path, path1 in combinations(chain_paths, 2):
    print(f"PMC({path}, {path1}) = {pairwise_mismatch_count(path, path1)}")
    print(f"PMC({path1}, {path}) = {pairwise_mismatch_count(path1, path)}")