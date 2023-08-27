from flask import Flask, render_template, request
import os
import speech_recognition as sr

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', converted_text='')

@app.route('/convert', methods=['POST'])
def convert():
    r = sr.Recognizer()

    if 'audio_file' not in request.files:
        return "No file part"

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        return "No selected file"

    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(audio_file_path)

    with sr.AudioFile(audio_file_path) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            return render_template('index.html', converted_text=text)
        except sr.UnknownValueError:
            return "Sorry, could not understand audio"
        except sr.RequestError as e:
            return f"Could not request results; {e}"

if __name__ == '__main__':
    app.run(debug=True)

