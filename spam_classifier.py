import os
import io
import numpy 
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            inBody = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
            f.close()
            message = '\n'.join(lines)
            yield path, message


def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    return DataFrame(rows, index=index)

data = DataFrame({'message': [], 'class': []})
data = data.append(dataFrameFromDirectory('/home/shasha/My Projects/Spam Classifier/emails/spam', 'spam'))
data = data.append(dataFrameFromDirectory('/home/shasha/My Projects/Spam Classifier/emails/ham', 'ham'))

# print(data.head())

#CountVectorizer counts all the words, converts them into tokens
vectorizer = CountVectorizer()
# How many times each word occurs in my email instead of storing each word we use a token
counts = vectorizer.fit_transform(data['message'].values)
classifier = MultinomialNB()
targets = data['class'].values
classifier.fit(counts, targets)

examples = ['Free Viagra now!!!', "Hi Bob, how about a game of golf tomorrow?"]
example_counts = vectorizer.transform(examples)
predictions = classifier.predict(example_counts)

i=0
while(i<len(examples)):
   print predictions[i] + ":",
   print examples[i]
   i+=1