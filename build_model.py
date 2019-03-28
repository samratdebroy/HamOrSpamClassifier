from collections import Counter
import re

# TASK 1: BUILD MODEL TEXT FILE

# Prepare Counters to count the frequency of each word in each dataset
ham_counter = Counter()
spam_counter = Counter()

# Prepare dict to track Conditional Probability for each word in each dataset
spam_cond_prob = Counter()
ham_cond_prob = Counter()

# Loop over all documents in the training set to obtain word frequency
num_ham_files = 1000
num_spam_files = 997

# Count the frequency of words in the Ham database
for ham_file_id in range(1, num_ham_files + 1):
    with open('Data/train/train-ham-{:05d}.txt'.format(ham_file_id),'r') as file:
        for line in file:
            # Get the words in this line
            words = re.split('[^a-zA-Z]', line.lower())
            # Count the frequency of each non-empty word
            non_empty_words = [word for word in words if word]
            ham_counter.update(non_empty_words)

# Count the frequency of words in the Spam database
for spam_file_id in range(1, num_spam_files + 1):
    with open('Data/train/train-spam-{:05d}.txt'.format(spam_file_id),'r') as file:
        for line in file:
            # Get the words in this line
            words = re.split('[^a-zA-Z]', line.lower())
            # Count the frequency of each non-empty word
            non_empty_words = [word for word in words if word]
            spam_counter.update(non_empty_words)

# count number of words per data base
num_spam_words = sum(spam_counter.values())
num_ham_words = sum(ham_counter.values())

# Get the vocabulary of words present in the Training Set
vocab = list(set(spam_counter).union(set(ham_counter)))
vocab.sort()

# Do smoothing and build conditional probabilities
delta = 0.5  # Smoothing amount per unique word
smoothed_spam_vocab_size = len(spam_counter)*delta
smoothed_ham_vocab_size = len(ham_counter)*delta
for word in vocab:
    spam_cond_prob[word] = (spam_counter[word] + delta)/(num_spam_words + smoothed_spam_vocab_size)
    ham_cond_prob[word] = (ham_counter[word] + delta)/(num_ham_words + smoothed_ham_vocab_size)

# Create and write the Model to file
with open('model.txt', 'w') as model_file:
    # Write each line of the model file
    for line_num, word in enumerate(vocab, start=1):
        line = '{num}  {word}  {ham_freq}  {ham_prob}  {spam_freq}  {spam_prob}\n'.format( num = line_num, word = word,
                                                    ham_freq = ham_counter[word], ham_prob = ham_cond_prob[word],
                                                    spam_freq = spam_counter[word], spam_prob = spam_cond_prob[word])
        model_file.write(line)
