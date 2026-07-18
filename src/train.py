import pandas as pd
import pickle
from preprocessing import clean_text
from tfidf import TFIDFVectorizer
from logistic_regression import LogisticRegression

data = pd.read_csv("data/spam.csv", encoding="latin-1")
data = data[["v1","v2"]]
data["text"] = data["v2"].apply(clean_text)
data["label"] = data["v1"].apply(lambda x: 1 if x=="spam" else 0)

split=int(len(data)*0.8)
train=data.iloc[:split]
test=data.iloc[split:]

vectorizer=TFIDFVectorizer(3000)
vectorizer.fit(train["text"].tolist())

X_train=vectorizer.transform(train["text"].tolist())
y_train=train["label"].tolist()

model=LogisticRegression()
model.train(X_train,y_train)

with open("models/vectorizer.pkl","wb") as f:
    pickle.dump(vectorizer,f)

with open("models/model.pkl","wb") as f:
    pickle.dump(model,f)
