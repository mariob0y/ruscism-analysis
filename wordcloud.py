import pandas as pd
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from natasha import Doc, Segmenter, NewsEmbedding, NewsMorphTagger
import pymorphy2
import re
morph = pymorphy2.MorphAnalyzer()
data_path = 'soloviev.json'

# Step 1: Load the JSON data
with open(data_path, encoding='utf-8') as f:
    data = json.load(f)

# Step 2: Load stopwords from the JSON file
with open('stopwords-ru.json', encoding='utf-8') as f:
    russian_stopwords = json.load(f)


root_word_map = {
    'российский': 'россия',
    'украинский': 'украина',
    'европейский': 'европа',
    "евросоюз": 'европа',
    "ес": "европа",
    "американский": "сша",
    "америка": "сша",
    "победить": "победа",
    "натовский": "нато",
    "польский" : "польша",
    "бандеровский":"бандера",
    "бандеровец": "бандера",
"нацистский":"нацист",
}

def lemmatize(text):
    words = text.split()
    res = list()
    for word in words:
        word = re.sub(r'[^а-яА-ЯёЁa-zA-Z]+', '', word)
        lemma = morph.parse(word)[0]
        lemma = lemma.normal_form
        if lemma in root_word_map:
            lemma = root_word_map[lemma]
        res.append(lemma)
    return res



# Step 4: Parse messages, lemmatize, and concatenate all texts
texts = []

for msg in data.get('messages', []):
    text = msg.get('text')
    if isinstance(text, str):

        lemmas = lemmatize(text)
        texts.append(' '.join(lemmas))

# Об'єднати всі тексти в один рядок
final_text = ' '.join(texts)

# Step 5: Generate the word cloud, ignoring stop words
wordcloud = WordCloud(width=1920, height=1080, background_color='white', colormap='viridis',
                      stopwords=russian_stopwords).generate(final_text)

# Step 6: Display the word cloud
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
