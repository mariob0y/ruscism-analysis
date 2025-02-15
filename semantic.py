from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import json
import string
import re
from tqdm import tqdm
import os
import glob
from utils import get_message_text, ukraine_mentioned
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
import matplotlib.dates as mdates
from scipy.interpolate import make_interp_spline
import numpy as np
from const import (
    dataset_path,
    semantic_chart_path,
    semantic_date_chart_path,
    semantic_output_path,
)


datasets = glob.glob(os.path.join(dataset_path, "*.json"))


def preprocess_data(text: str) -> str:
    text = re.sub(r"[^а-яА-ЯёЁa-zA-Z]+", " ", text)
    return text.lower().translate(str.maketrans("", "", string.punctuation)).strip()


def is_sarcastic(text):
    tokenized_text = sarcasm_tokenizer(
        [preprocess_data(text)],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    output = sarcasm_classifier(**tokenized_text)
    probs = output.logits.softmax(dim=-1).tolist()[0]
    confidence = max(probs)
    return probs.index(confidence)


sentiment_model = "seara/rubert-tiny2-russian-sentiment"
sentiment_classifier = pipeline("sentiment-analysis", model=sentiment_model)

sarcasm_model = "helinivan/multilingual-sarcasm-detector"
sarcasm_tokenizer = AutoTokenizer.from_pretrained(sarcasm_model)
sarcasm_classifier = AutoModelForSequenceClassification.from_pretrained(sarcasm_model)


def generate_pie_chart(topic_count, semantic_count, dataset_name):
    def filter_zeros(labels, sizes, colors):
        filtered = [
            (label, size, color)
            for label, size, color in zip(labels, sizes, colors)
            if size > 0
        ]
        if not filtered:
            return ["No Data"], [1], ["#d3d3d3"]
        return zip(*filtered)

    labels_topic, sizes_topic, colors_topic = filter_zeros(
        ["Україна", "інша"],
        [topic_count["ukraine"], topic_count["rest"]],
        ["#66b3ff", "#ff9999"],
    )

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.pie(
        sizes_topic,
        labels=labels_topic,
        autopct="%1.1f%%" if len(labels_topic) > 1 else None,
        colors=colors_topic,
        startangle=90,
    )
    plt.title("Розподіл за темами", pad=20)
    plt.figtext(
        0.25,
        0.01,
        f"Загальна кількість: {topic_count['total']}",
        ha="center",
        fontsize=12,
    )

    labels_semantic, sizes_semantic, colors_semantic = filter_zeros(
        ["позитивне", "нейтральне", "негативне"],
        [
            semantic_count["positive"],
            semantic_count["neutral"],
            semantic_count["negative"],
        ],
        ["#99ff99", "#ffcc99", "#ff6666"],
    )

    plt.subplot(1, 2, 2)
    plt.pie(
        sizes_semantic,
        labels=labels_semantic,
        autopct="%1.1f%%" if len(labels_semantic) > 1 else None,
        colors=colors_semantic,
        startangle=90,
    )
    plt.title("Розподіл емоційного забарвлення", pad=20)
    plt.figtext(
        0.75,
        0.01,
        f"Загальна кількість: {semantic_count['total']}",
        ha="center",
        fontsize=12,
    )

    plt.suptitle(f"Семантичний аналіз {dataset_name}", fontsize=14)
    plt.tight_layout()

    plt.savefig(
        os.path.join(semantic_chart_path, f"{dataset_name}_semantic_chart.png"),
        bbox_inches="tight",
    )
    plt.close()


def process_semantic_messages(data, dataset_name):
    messages = data.get("messages", [])
    topic_count = {"ukraine": 0, "rest": 0, "total": 0}
    semantic_count = {"positive": 0, "neutral": 0, "negative": 0, "total": 0}
    date_count = defaultdict(int)

    total_messages = len(messages)
    for msg in tqdm(messages, desc=f"Обробка {dataset_name}", total=total_messages):
        msg_type = msg.get("type")
        if msg_type != "message":
            continue
        text = get_message_text(msg)

        if not text:
            continue

        topic_count["total"] += 1

        msg_date = datetime.fromtimestamp(
            int(msg["date_unixtime"]), timezone.utc
        ).strftime("%Y-%m-%d")

        # Topic check
        if not ukraine_mentioned(text):
            topic_count["rest"] += 1
            # accessing key so that 0 would be written in dict
            date_count[msg_date]
            continue
        topic_count["ukraine"] += 1

        date_count[msg_date] += 1

        # Sentiment
        sentiment = sentiment_classifier(text)
        s_label = sentiment[0]["label"].lower()

        if s_label != "negative" and is_sarcastic(text):
            semantic_count["negative"] += 1
        else:
            semantic_count[s_label] += 1
        semantic_count["total"] += 1

    return topic_count, semantic_count, date_count


def generate_date_chart(data, dataset_name):
    dates = list(data.keys())
    counts = list(data.values())

    dates = [pd.to_datetime(date) for date in dates]

    plt.figure(figsize=(10, 5))

    x = mdates.date2num(dates)
    spline = make_interp_spline(x, counts, k=3)
    x_new = np.linspace(x.min(), x.max(), 500)
    y_new = spline(x_new)

    plt.plot_date(dates, counts, "b-", linestyle="-", label="Згадки України")

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gca().xaxis.set_minor_locator(mdates.WeekdayLocator())

    plt.xticks(rotation=45)

    plt.xlabel("Дата")
    plt.ylabel("Кількість згадок України")

    plt.title(f"Часова діаграма {dataset_name}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            semantic_date_chart_path, f"{dataset_name}_semantic_date_chart.png"
        ),
        bbox_inches="tight",
    )
    plt.close()


def main():
    for file_path in datasets:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]

        result_path = f"{semantic_output_path}/{dataset_name}.json"
        if os.path.exists(result_path):
            continue
        print(f"Processing {dataset_name}...")
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        topic_count, semantic_count, date_count = process_semantic_messages(
            data, dataset_name
        )
        generate_pie_chart(topic_count, semantic_count, dataset_name)
        generate_date_chart(date_count, dataset_name)
        with open(result_path, "w", encoding="utf-8") as json_file:
            json.dump(
                [topic_count, semantic_count, date_count],
                json_file,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
