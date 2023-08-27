from langdetect import detect
from mtranslate import translate

# Simple function to detect and translate text
def detect_and_translate(input_text=None, target_lang=None):
    # If input_text and target_lang are provided, use them, else prompt user
    if input_text is None:
        input_text = input("Enter the text to translate: ")
    if target_lang is None:
        target_lang = input("Enter the target language: ")

    result_lang = detect(input_text)

    if result_lang == target_lang:
        return input_text
    else:
        translate_text = translate(input_text, target_lang)
        return translate_text

if __name__ == "__main__":
    translated_text = detect_and_translate()
    print("Translated text:", translated_text)
