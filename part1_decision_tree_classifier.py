import os
import pandas as pd
import math

# Load the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "iris.csv")
df = pd.read_csv(file_path)

# Entropy Calculation
def calculate_entropy(data):
    labels = data['variety']
    total = len(labels)
    label_counts = labels.value_counts()

    entropy = 0
    for count in label_counts:
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# Information Gain Calculation
def information_gain(data, attribute):
    total_entropy = calculate_entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        weight = len(subset) / len(data)
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += weight * subset_entropy

    return total_entropy - weighted_entropy

# Decision Tree (Split=True)
def build_tree(data, depth=0):
    labels = data['variety'].unique()

    if len(labels) == 1:
        return {'label': labels[0]}

    if data.empty:
        return {'label': 'Unknown'}

    best_gain = 0
    best_attr = None
    for attr in data.columns[:-1]:
        gain = information_gain(data, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr

    if best_gain == 0 or best_attr is None:
        return {'label': data['variety'].mode()[0]}

    tree = {'attribute': best_attr, 'nodes': {}}
    values = data[best_attr].unique()

    for value in values:
        subset = data[data[best_attr] == value]
        subtree = build_tree(subset, depth + 1)
        tree['nodes'][value] = subtree

    return tree

# Tree Printer
def print_tree(tree,):
    if "label" in tree:
        print(tree["label"])
    else:
        print( f" Split: {tree['attribute']}")
        for value, subtree in tree["nodes"].items():
            print( f"  └─ {tree['attribute']} = {value}")
            print_tree(subtree)

# No Split Classifier
def no_split_classifier(data):
    return data['variety'].mode()[0]

# Execution Section
print("\n--- Initial Data Overview ---")
print(df.head())
print("\nClass Distribution:")
print(df['variety'].value_counts())

print("\nEntropy of Dataset:")
print(f"{calculate_entropy(df):.4f}")

print("\nInformation Gain for each attribute:")
for col in df.columns[:-1]:
    print(f"{col}: {information_gain(df, col):.4f}")

print("\n--- Decision Tree (Split=True) ---")
tree = build_tree(df)
print_tree(tree)

print("\n--- No-Split Classifier ---")
prediction = no_split_classifier(df)
print(f"Prediction (most common class): {prediction}")
