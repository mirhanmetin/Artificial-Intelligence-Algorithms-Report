import pandas as pd
import random
import math
from collections import defaultdict, Counter

# Load cleaned data
df = pd.read_excel("/Users/mirhanmetin/CMP4501/CMP_code_assignment_P2/part2_naive_bayes_classifier/processed_comments34-2.xlsx", engine="openpyxl")

# Use lemmatized text and labels
texts = df['Lemmatized'].astype(str).tolist()
labels = df['Label'].astype(str).tolist()

# Split into train and test sets
combined = list(zip(texts, labels))
random.shuffle(combined)
split_point = int(0.8 * len(combined))
train_data = combined[:split_point]
test_data = combined[split_point:]

# Training Phase
class_word_counts = defaultdict(Counter)
class_counts = Counter()
total_words_per_class = defaultdict(int)

for text, label in train_data:
    words = text.split()
    class_counts[label] += 1
    for word in words:
        class_word_counts[label][word] += 1
        total_words_per_class[label] += 1

vocab = set(word for class_dict in class_word_counts.values() for word in class_dict)
vocab_size = len(vocab)

def predict(text):
    words = text.split()
    class_scores = {}
    total_docs = sum(class_counts.values())

    for label in class_counts:
        log_prob = math.log(class_counts[label] / total_docs)
        for word in words:
            word_freq = class_word_counts[label][word]
            word_prob = (word_freq + 1) / (total_words_per_class[label] + vocab_size)
            log_prob += math.log(word_prob)
        class_scores[label] = log_prob

    return max(class_scores, key=class_scores.get)

# Evaluation
correct = 0
total = len(test_data)

for text, label in test_data:
    prediction = predict(text)
    if prediction == label:
        correct += 1

accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

# Show example predictions
print("\nSample predictions:")
for i in range(5):
    text, label = test_data[i]
    pred = predict(text)
    print(f"Text: {text[:60]}...\nActual: {label}, Predicted: {pred}\n")
