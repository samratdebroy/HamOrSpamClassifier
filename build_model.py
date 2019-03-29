from collections import Counter
import re
import math

def get_word_frequency():
    # Prepare Counters to count the frequency of each word in each dataset
    ham_counter = Counter()
    spam_counter = Counter()

    # Loop over all documents in the training set to obtain word frequency
    num_ham_files = 1000
    num_spam_files = 997

    # Get probability that an email is either spam or ham
    ham_prob = num_ham_files / (num_ham_files + num_spam_files)
    spam_prob = num_spam_files / (num_ham_files + num_spam_files)

    # Count the frequency of words in the Ham database
    for ham_file_id in range(1, num_ham_files + 1):
        with open('Data/train/train-ham-{:05d}.txt'.format(ham_file_id),'r', encoding='latin-1') as file:
            for line in file:
                # Get the words in this line
                words = re.split('[^a-zA-Z]', line.lower())
                # Count the frequency of each non-empty word
                non_empty_words = [word for word in words if word]
                ham_counter.update(non_empty_words)

    # Count the frequency of words in the Spam database
    for spam_file_id in range(1, num_spam_files + 1):
        with open('Data/train/train-spam-{:05d}.txt'.format(spam_file_id),'r', encoding='latin-1') as file:
            for line in file:
                # Get the words in this line
                words = re.split('[^a-zA-Z]', line.lower())
                # Count the frequency of each non-empty word
                non_empty_words = [word for word in words if word]
                spam_counter.update(non_empty_words)

    return spam_counter, ham_counter, spam_prob, ham_prob

def create_model(spam_counter, ham_counter, model_filename = 'model.txt'):

    # Get the vocabulary of words present in the Training Set
    vocab = list(set(spam_counter).union(set(ham_counter)))
    vocab.sort()

    # count number of words per data base
    num_spam_words = sum(spam_counter.values())
    num_ham_words = sum(ham_counter.values())

    # Prepare Counters to track Conditional Probability for each word in each dataset
    spam_cond_prob = Counter()
    ham_cond_prob = Counter()

    # Do smoothing and build conditional probabilities
    delta = 0.5  # Smoothing amount per unique word
    smoothed_spam_vocab_size = len(spam_counter)*delta
    smoothed_ham_vocab_size = len(ham_counter)*delta
    for word in vocab:
        spam_cond_prob[word] = (spam_counter[word] + delta)/(num_spam_words + smoothed_spam_vocab_size)
        ham_cond_prob[word] = (ham_counter[word] + delta)/(num_ham_words + smoothed_ham_vocab_size)

    # Create and write the Model to file
    with open(model_filename, 'w', encoding='latin-1') as model_file:
        # Write each line of the model file
        for line_num, word in enumerate(vocab, start=1):
            line = '{num}  {word}  {ham_freq}  {ham_prob}  {spam_freq}  {spam_prob}\n'.format( num = line_num,
                                            word = word, ham_freq = ham_counter[word], ham_prob = ham_cond_prob[word],
                                            spam_freq = spam_counter[word], spam_prob = spam_cond_prob[word])
            model_file.write(line)

    return vocab, ham_cond_prob, spam_cond_prob

def evaluate_model(ham_cond_prob, spam_cond_prob, spam_prob, ham_prob, results_filename = 'result.txt'):

    num_ham_files = 400
    num_spam_files = 400

    # counters for correct classification
    ham_correct = 0
    spam_correct = 0

    # Create and write the Model to file
    with open(results_filename, 'w', encoding='latin-1') as model_file:

        # Loop through each file in the Test Ham database
        for ham_file_id in range(1, num_ham_files + 1):
            filename = 'test-ham-{:05d}.txt'.format(ham_file_id)
            ham_score, spam_score, email_type = classify('Data/test/{}'.format(filename), ham_cond_prob,
                                                         spam_cond_prob, spam_prob, ham_prob)

            if email_type == 'ham':
                ham_correct += 1
        
            # Write each line of the model file
            line = '{num}  {name}  {result}  {ham_score}  {spam_score}  {real_type}  {correct}\n'.format(
                num=ham_file_id, name=filename, result=email_type, ham_score=ham_score, spam_score=spam_score,
                real_type='ham',correct=(email_type == 'ham'))
            model_file.write(line)

        # Loop through each file in the Test Ham database
        for spam_file_id in range(1, num_spam_files + 1):
            filename = 'test-spam-{:05d}.txt'.format(spam_file_id)
            ham_score, spam_score, email_type = classify('Data/test/{}'.format(filename), ham_cond_prob,
                                                         spam_cond_prob, spam_prob, ham_prob)

            if email_type == 'spam':
                spam_correct += 1

            # Write each line of the model file
            line = '{num}  {name}  {result}  {ham_score}  {spam_score}  {real_type}  {correct}\n'.format(
                num=spam_file_id + num_ham_files, name=filename, result=email_type, ham_score=ham_score,
                spam_score=spam_score, real_type='spam',correct=(email_type == 'spam'))
            model_file.write(line)

        # number of incorrectly classified emails
        ham_incorrect = num_spam_files - spam_correct
        spam_incorrect = num_ham_files - ham_correct

        total_correct = ham_correct + spam_correct

        accuracy = total_correct / (total_correct + ham_incorrect + spam_incorrect)
        precision = ham_correct / (ham_correct + ham_incorrect)
        recall = ham_correct / (ham_correct + spam_incorrect)
        f_measure = (2 * precision * recall) / (precision + recall)

        print('Error Analysis:\n Accuracy: {}\n Precision: {}\n Recall: {}\n F-Measure: {}\n'.format(
            accuracy, precision, recall, f_measure))


def classify(email_filename, ham_cond_prob, spam_cond_prob, spam_prob, ham_prob):
    ham_score = math.log10(ham_prob)
    spam_score = math.log10(spam_prob)

    with open(email_filename, 'r', encoding='latin-1') as file:
        for line in file:
            # Get the words in this line
            words = re.split('[^a-zA-Z]', line.lower())

            # Add the score from each word
            non_empty_words = [word for word in words if word]
            for word in non_empty_words:
                if ham_cond_prob[word]:
                    ham_score += math.log10(ham_cond_prob[word])
                if spam_cond_prob[word]:
                    spam_score += math.log10(spam_cond_prob[word])

    if ham_score > spam_score:
        email_type = 'ham'
    else:
        email_type = 'spam'
    return ham_score, spam_score, email_type
