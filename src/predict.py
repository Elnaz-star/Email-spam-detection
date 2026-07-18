import pickle
from preprocessing import clean_text

with open("models/vectorizer.pkl","rb") as f:
    vectorizer=pickle.load(f)

with open("models/model.pkl","rb") as f:
    model=pickle.load(f)

email=input("Email: ")
text=clean_text(email)
vector=vectorizer.transform([text])

print("Spam" if model.predict(vector)[0]==1 else "Ham")
