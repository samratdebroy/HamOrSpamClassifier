import build_model

def experiment1():
    # Parse training data to build word frequencies
    spam_counter, ham_counter, spam_prob, ham_prob = build_model.get_word_frequency()

    # Create model of conditional probabilities
    vocab, ham_cond_prob, spam_cond_prob = build_model.create_model(spam_counter, ham_counter,
                                                                    model_filename="model.txt")

    # Evaluate the effectiveness of the model over the test set
    build_model.evaluate_model(ham_cond_prob, spam_cond_prob, spam_prob, ham_prob,
                               results_filename='baseline-result.txt')


def experiment2():
    # Parse training data to build word frequencies
    spam_counter, ham_counter, spam_prob, ham_prob = build_model.get_word_frequency()

    # Get the stopwords
    ham_words = sorted(ham_counter.keys())
    spam_words = sorted(spam_counter.keys())
    with open('Data/English-Stop-Words.txt', 'r', encoding='latin-1') as file:
        for stopword in file:
            # Remove stopwords from the counters
            stopword = stopword[:-1]  # remove \n
            if stopword in ham_words:
                del ham_counter[stopword]
            if stopword in spam_words:
                del spam_counter[stopword]

    # Create model of conditional probabilities
    vocab, ham_cond_prob, spam_cond_prob = build_model.create_model(spam_counter, ham_counter,
                                                                    model_filename="stopword-model.txt")

    # Evaluate the effectiveness of the model over the test set
    build_model.evaluate_model(ham_cond_prob, spam_cond_prob, spam_prob, ham_prob,
                               results_filename='stopword-result.txt')

def experiment3():
    # Parse training data to build word frequencies
    spam_counter, ham_counter, spam_prob, ham_prob = build_model.get_word_frequency()

    # Get the vocabulary of words present in the Training Set
    vocab = list(set(spam_counter).union(set(ham_counter)))
    vocab.sort()

    # Check if the word is the appropriate length
    ham_words = sorted(ham_counter.keys())
    spam_words = sorted(spam_counter.keys())
    for word in vocab:
        if len(word) <= 2 or  len(word) >= 9:
            # Remove words of bad length from the counters
            if word in ham_words:
                del ham_counter[word]
            if word in spam_words:
                del spam_counter[word]

    # Create model of conditional probabilities
    vocab, ham_cond_prob, spam_cond_prob = build_model.create_model(spam_counter, ham_counter,
                                                                    model_filename="wordlength-model.txt")

    # Evaluate the effectiveness of the model over the test set
    build_model.evaluate_model(ham_cond_prob, spam_cond_prob, spam_prob, ham_prob,
                               results_filename='wordlength-result.txt')


if __name__ == '__main__':
    experiment3()
