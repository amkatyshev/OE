import re


def clean(text: str) -> str:
    result = re.sub(r"[^\sа-яА-ЯёЁ0-9.,*_!?\\\"\'\(\)\-\+=–]", "", text)
    # result = re.sub("[a-zA-Z]", "", text)
    result = re.sub(r"\s{2,}", " ", result)  # Повторящиеся пробелы
    result = re.sub(r"(\d)([а-яА-ЯёЁ])", r"\1 \2", result)  # Отступы между цифрой и буквой
    result = re.sub("\n", " ", result)  # Перевод строки
    result = re.sub(r"([.,*_!?\\\"\'\)\-\+=–])(\w)", r"\1 \2", result)
    return result