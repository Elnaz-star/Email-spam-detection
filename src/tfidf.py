import math

class TFIDFVectorizer:
    def __init__(self, max_features=3000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf = {}

    def fit(self, documents):
        count = {}
        for doc in documents:
            words = set(doc.split())
            for word in words:
                count[word] = count.get(word, 0) + 1

        words = sorted(count, key=count.get, reverse=True)[:self.max_features]
        self.vocabulary = {w:i for i,w in enumerate(words)}

        n = len(documents)
        for word in self.vocabulary:
            self.idf[word] = math.log((n + 1) / (count[word] + 1)) + 1

    def transform(self, documents):
        result = []
        for doc in documents:
            row = [0.0] * len(self.vocabulary)
            words = doc.split()
            total = len(words) if words else 1
            for word in words:
                if word in self.vocabulary:
                    tf = words.count(word) / total
                    row[self.vocabulary[word]] = tf * self.idf[word]
            result.append(row)
        return result
