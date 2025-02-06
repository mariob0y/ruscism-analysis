import json
import os
import glob
import re
import pymorphy2
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import get_message_text

# Initialize the morphological analyzer for Russian
morph = pymorphy2.MorphAnalyzer()

# Load Russian stopwords from a JSON file
with open("stopwords-ru.json", encoding="utf-8") as f:
    russian_stopwords = set(json.load(f))  # Convert to set for faster lookup

# Define paths for datasets and output directories
dataset_path = "datasets"
diagram_path = "diagrams"
wordclouds_path = "wordclouds"

# Ensure output directories exist
os.makedirs(diagram_path, exist_ok=True)
os.makedirs(wordclouds_path, exist_ok=True)

# Mapping words to their root equivalents to normalize variations
root_word_map = {
    "рф": "россия",
    "российский": "россия",
    "украинский": "украина",
    "европейский": "европа",
    "евросоюз": "европа",
    "ес": "европа",
    "американский": "сша",
    "америка": "сша",
    "победить": "победа",
    "натовский": "нато",
    "польский": "польша",
    "бандеровский": "бандера",
    "бандеровец": "бандера",
    "нацистский": "нацист",
}


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
    )[
        :50
    ]  # Get top 50 words

    if not sorted_word_count:
        print(f"No significant words found for {dataset_name}, skipping chart.")
        return

    words, frequencies = zip(*sorted_word_count)

    plt.figure(figsize=(10, 15))
    plt.barh(words, frequencies, color="skyblue", height=0.4)
    plt.xlabel("Частота", fontsize=12)
    plt.ylabel("Слова", fontsize=12)
    plt.title(f"Частота слів у тексті {dataset_name} (Топ-50)", fontsize=14)
    plt.gca().invert_yaxis()

    for i, v in enumerate(frequencies):
        plt.text(v + 1, i, str(v), fontsize=10, verticalalignment="center")

    plt.savefig(
        os.path.join(diagram_path, f"{dataset_name}_chart.png"), bbox_inches="tight"
    )
    plt.close()


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
        os.path.join(wordclouds_path, f"{dataset_name}_wordcloud.png"),
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

        create_chart(wordcloud, final_text, dataset_name)
        create_wordcloud(wordcloud, final_text, dataset_name)
        print(f"Finished processing {dataset_name}.")


if __name__ == "__main__":
    main()
