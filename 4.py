from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from bertopic import BERTopic
import json
import string
import re
from tqdm import tqdm  # Для індикатора прогресу

# Завантаження даних
with open('medvedev.json', encoding='utf-8') as f:
    data = json.load(f)

# Функція для попередньої обробки тексту
def preprocess_data(text: str) -> str:
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z]+', ' ', text)
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()

# Ініціалізація pipeline для сентименту та завантаження моделі сарказму
sentiment_model = 'seara/rubert-tiny2-russian-sentiment'
sentiment_classifier = pipeline("sentiment-analysis", model=sentiment_model)

sarcasm_model = "helinivan/multilingual-sarcasm-detector"
sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model)
sarcasm_classifier = AutoModelForSequenceClassification.from_pretrained(sarcasm_model)

# Лічильники для сентименту та сарказму
semantic_count = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
sarcastic_count = {'sarcastic': 0, 'not': 0, 'total': 0}

# Збір та попередня обробка текстів для аналізу тем
messages = data.get('messages', [])
total_messages = len(messages)
documents = []

for msg in tqdm(messages, desc="Обробка повідомлень", total=total_messages):
    msg_type = msg.get('type')
    if msg_type != 'message':
        continue
    text = msg.get('text')

    if text and isinstance(text, str):
        preprocessed_text = preprocess_data(text)
        documents.append(preprocessed_text)

        # Сентимент-аналіз
        sentiment = sentiment_classifier(text)
        s_label = sentiment[0]['label'].lower()

        # Сарказм-аналіз
        tokenized_text = sarcasm_tokenizer([preprocessed_text], padding=True, truncation=True, max_length=256,
                                           return_tensors="pt")
        output = sarcasm_classifier(**tokenized_text)
        probs = output.logits.softmax(dim=-1).tolist()[0]
        confidence = max(probs)
        is_sarcastic = probs.index(confidence)

        # Оновлення лічильників
        if is_sarcastic:
            sarcastic_count['sarcastic'] += 1
        else:
            sarcastic_count['not'] += 1

        if s_label in ['positive', 'neutral']:
            if is_sarcastic:
                semantic_count['negative'] += 1
            else:
                semantic_count[s_label] += 1
        else:
            semantic_count[s_label] += 1
        semantic_count['total'] += 1
        sarcastic_count['total'] += 1

# Виведення результатів сентимент-аналізу та сарказму
print("Семантичний аналіз:", semantic_count)
print("Аналіз на сарказм:", sarcastic_count)

# Аналіз тем за допомогою BERTopic
print("\nАналіз тем за допомогою BERTopic:")
topic_model = BERTopic(language="multilingual")  # Модель для багатомовного аналізу
topics, _ = topic_model.fit_transform(documents)

# Отримання інформації про теми
topics_info = topic_model.get_topic_info()
print(topics_info)

# Перегляд ключових слів для кожної теми
for topic in topics_info['Topic']:
    if topic != -1:  # -1 позначає тексти, що не підпали під певну тему
        print(f"Тема {topic}:")
        print(topic_model.get_topic(topic))
