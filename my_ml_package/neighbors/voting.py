from collections import Counter

def majority_vote(labels):
    counter = Counter(labels)

    return counter.most_common(1)[0][0]