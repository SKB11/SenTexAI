from langdetect import detect
from mtranslate import translate

# Simple function to detect and translate text
def detect_and_translate():
    # Prompt the user for input
    text = input("Enter the text to translate: ")
    target_lang = input("Enter the target language: ")

    result_lang = detect(text)

    if result_lang == target_lang:
        return text
    else:
        translate_text = translate(text, target_lang)
        return translate_text

translated_text = detect_and_translate()
print("Translated text:", translated_text)