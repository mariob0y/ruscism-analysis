import re
import pymorphy2
from const import root_word_map, ukraine_keywords

import inspect

# Patch inspect.getargspec to use inspect.getfullargspec properly
if not hasattr(inspect, "getargspec"):

    def getargspec_patched(func):
        spec = inspect.getfullargspec(func)
        return spec.args, spec.varargs, spec.varkw, spec.defaults

    inspect.getargspec = getargspec_patched  # Monkey-patch for compatibility

import pymorphy2

# Now it should work correctly
morph = pymorphy2.MorphAnalyzer()


def get_message_text(msg):
    msg_type = msg.get("type")
    if msg_type != "message":
        return  # Ignore service messages
    if msg.get("forwarded_from"):
        return  # Ignore forwarded messages

    message_text = ""
    text_data = msg.get("text")

    if isinstance(text_data, str):
        message_text = text_data
    elif isinstance(text_data, list):
        for item in text_data:
            if isinstance(item, str):
                message_text += f" {item}"
            elif isinstance(item, dict) and item.get("type") not in [
                "link",
                "text_link",
            ]:
                message_text += f" {item.get('text', '')}"
    return message_text


def lemmatize(text):
    """
    Tokenizes and lemmatizes the input text, mapping words to their root forms.
    """
    words = re.split(r"[\n \-_,.]+", text)  # Split by common delimiters
    res = []
    for word in words:
        if word.startswith("#"):
            continue  # Ignore hashtags
        word = re.sub(
            r"[^а-яА-ЯёЁa-zA-Z]+", "", word
        )  # Remove non-alphabetic characters
        if not word:
            continue
        lemma = morph.parse(word)[0].normal_form  # Get the normal form (lemma)
        lemma = root_word_map.get(
            lemma, lemma
        )  # Replace with mapped root word if applicable
        res.append(lemma)
    return res


def ukraine_mentioned(text):
    lemmas = lemmatize(text)
    if any(lemma in ukraine_keywords for lemma in lemmas):
        return True
    return False
