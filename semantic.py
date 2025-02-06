from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import string
import re
from tqdm import tqdm  # Імпортуємо tqdm для індикатора прогресу
import os
import glob
from utils import get_message_text

dataset_path = "datasets"
datasets = glob.glob(os.path.join(dataset_path, "*.json"))


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

zero_shot_clasifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
hypotheses = [
    "This text is about Ukraine",
    "This text mentions Ukraine",
    "This text mentions the enemy",
    "This text is about the Armed Forces of Ukraine",
    "This text is about opponents of Russia",
    "This text refers to war in Ukraine",
]



for file_path in datasets:
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Processing {dataset_name}...")
    with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
    messages = data.get('messages', [])
    total_messages = len(messages)

    topic_count = {'ukraine': 0, 'rest': 0, 'total': 0}
    semantic_count = {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}

    for msg in tqdm(messages, desc=f"Обробка {dataset_name}", total=total_messages):
        msg_type = msg.get('type')
        if msg_type != 'message':
            continue
        text = get_message_text(msg)

        if not text:
            continue

        topic_count['total'] += 1

        # Topic check
        zero_shot = zero_shot_clasifier(text, hypotheses, multi_label=True)
        if not any(zero_shot['scores']) > 0.7:
            topic_count['rest'] += 1
            continue

        topic_count['ukraine'] += 1


        # Сентимент
        sentiment = sentiment_classifier(text)
        s_label = sentiment[0]['label'].lower()

        # Сарказм
        tokenized_text = sarcasm_tokenizer([preprocess_data(text)], padding=True, truncation=True, max_length=256,
                                   return_tensors="pt")
        output = sarcasm_classifier(**tokenized_text)
        probs = output.logits.softmax(dim=-1).tolist()[0]
        confidence = max(probs)
        is_sarcastic = probs.index(confidence)

        if s_label in ['positive', 'neutral']:
            if is_sarcastic:
                semantic_count['negative'] += 1
            else:
                semantic_count[s_label] += 1
        else:
            semantic_count[s_label] += 1
        semantic_count['total'] += 1

    # Виведення результатів
    print(semantic_count)
    print(topic_count)
