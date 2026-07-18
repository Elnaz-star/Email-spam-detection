import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\S+@\S+", " emailaddress ", text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = " ".join(text.split())
    return text
