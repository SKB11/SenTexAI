import speech_recognition as sr

r = sr.Recognizer()

# Update the audio file path with double backslashes
audio_file_path = r'C:\Users\Sharddha K B\SenTexAI-main\Data\Recordings\Recording.wav'

with sr.AudioFile(audio_file_path) as source:
    audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("Working on...")
        print(text)
    except sr.UnknownValueError:
        print("Sorry, could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
