import json
import os
import glob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import get_message_text, lemmatize
from const import (
    dataset_path,
    wordclouds_chart_path,
    word_count_chart_path,
    word_count_output_path,
)


with open("stopwords-ru.json", encoding="utf-8") as f:
    russian_stopwords = set(json.load(f))


def process_dataset(data):
    """
    Extracts text messages, lemmatizes them, and concatenates them into a single string.
    """
    texts = []
    for msg in data.get("messages", []):
        message_text = get_message_text(msg)
        if message_text:
            lemmas = lemmatize(message_text)
            texts.append(" ".join(lemmas))

    return " ".join(texts)


def create_chart(wordcloud, final_text, dataset_name):
    """
    Generates and saves a horizontal bar chart of the most frequent words.
    """
    word_count = wordcloud.process_text(final_text)
    sorted_word_count = sorted(
        (
            (word, count)
            for word, count in word_count.items()
            if word not in russian_stopwords
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    if not sorted_word_count:
        print(f"No significant words found for {dataset_name}, skipping chart.")
        return

    words, frequencies = zip(*sorted_word_count[:50])

    plt.figure(figsize=(10, 15))
    plt.barh(words, frequencies, color="skyblue", height=0.4)
    plt.xlabel("Частота", fontsize=12)
    plt.ylabel("Слова", fontsize=12)
    plt.title(f"Частота слів у тексті {dataset_name} (Топ-50)", fontsize=14)
    plt.gca().invert_yaxis()

    for i, v in enumerate(frequencies):
        plt.text(v + 1, i, str(v), fontsize=10, verticalalignment="center")

    plt.savefig(
        os.path.join(word_count_chart_path, f"{dataset_name}_chart.png"),
        bbox_inches="tight",
    )
    plt.close()
    return sorted_word_count


def create_wordcloud(wordcloud, final_text, dataset_name):
    """
    Generates and saves a word cloud image.
    """
    if not final_text.strip():
        print(f"No text available for {dataset_name}, skipping word cloud.")
        return

    visual_result = wordcloud.generate(final_text)
    plt.figure(figsize=(10, 20))
    plt.imshow(visual_result, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(
        os.path.join(wordclouds_chart_path, f"{dataset_name}_wordcloud.png"),
        bbox_inches="tight",
    )
    plt.close()


def main():
    """
    Main function: Processes all JSON datasets and generates word clouds and charts.
    """
    datasets = glob.glob(os.path.join(dataset_path, "*.json"))

    if not datasets:
        print("No datasets found in the folder.")
        return

    for file_path in datasets:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = f"{word_count_output_path}/{dataset_name}.json"

        if os.path.exists(output_path):
            continue
        print(f"Processing {dataset_name}...")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        final_text = process_dataset(data)

        wordcloud = WordCloud(
            height=1920,
            width=1080,
            background_color="white",
            colormap="viridis",
            stopwords=russian_stopwords,
        )

        word_count = create_chart(wordcloud, final_text, dataset_name)
        create_wordcloud(wordcloud, final_text, dataset_name)
        print(f"Finished processing {dataset_name}.")
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(
                word_count,
                json_file,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()
