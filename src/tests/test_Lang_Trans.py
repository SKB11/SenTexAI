from pages.Lang_Trans import detect_and_translate

def test_same_language():
    text = "Hello"
    target_lang = "en"
    translated_text = detect_and_translate(text, target_lang)
    assert translated_text == text
    print("Test 'test_same_language' passed successfully!")

def test_translation():
    text = "Bonjour"
    target_lang = "en"
    translated_text = detect_and_translate(text, target_lang)
    assert translated_text != text
    assert translated_text != ""
    print("Test 'test_translation' passed successfully!")

def test_empty_input():
    text = ""
    target_lang = "es"
    translated_text = detect_and_translate(text, target_lang)
    assert translated_text == ""
    print("Test 'test_empty_input' passed successfully!")

def test_invalid_language():
    text = "Hello"
    target_lang = "invalid"
    translated_text = detect_and_translate(text, target_lang)
    assert translated_text == ""
    print("Test 'test_invalid_language' passed successfully!")
