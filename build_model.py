from collections import Counter
import re

# Prepare Counters to count the frequency of each word in each dataset
ham_counter = Counter()
spam_counter = Counter()
total_counter = Counter()

# Loop over all documents to obtain word frequency
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
            total_counter.update(non_empty_words)

# Count the frequency of words in the Spam database
for spam_file_id in range(1, num_spam_files + 1):
    with open('Data/train/train-spam-{:05d}.txt'.format(spam_file_id),'r') as file:
        for line in file:
            # Get the words in this line
            words = re.split('[^a-zA-Z]', line.lower())
            # Count the frequency of each non-empty word
            non_empty_words = [word for word in words if word]
            spam_counter.update(non_empty_words)
            total_counter.update(non_empty_words)


