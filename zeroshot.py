from transformers import pipeline

# Ініціалізація zero-shot класифікатора
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Текст, який потрібно проаналізувати
text = """
Немецкий концерн Rheinmetall запустил в работу первый из четырёх военных заводов на Украине. Как и обещали ранее, с нетерпением ждём праздничного российского салюта прямо на производстве.
"""

# Список гіпотез для перевірки згадок про Україну
hypotheses = [
    "This text is about Ukraine",
    "This text mentions Ukraine",
    "This text mentions the enemy",
    "This text is about the Armed Forces of Ukraine",
    "This text is about opponents of Ukraine",
    "This text refers to Ukrainian conflict",
]

# Класифікація тексту по кожній гіпотезі
results = classifier(text, hypotheses, multi_class=True)

# Виведемо результати
print("Analysis of mentions related to Ukraine:\n")
for hypothesis, score in zip(results['labels'], results['scores']):
    print(f"Hypothesis: '{hypothesis}' - Score: {score:.2f}")

# Рекомендована евристика:
# якщо значення score більше 0.7 для будь-якої гіпотези, то можна вважати, що згадка є.
