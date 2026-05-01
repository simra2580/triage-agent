import re

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    return re.sub(r'[^a-z0-9\s]', '', text)