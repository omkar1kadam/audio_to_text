from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def transcribe_with_whisper_cpp(audio_path):
    txt_output = audio_path.rsplit('.', 1)[0] + ".txt"

    # Call whisper.cpp
    command = [
        "./build/bin/whisper-cli",
        "-m", "models/ggml-base.bin",
        "-f", audio_path,
        "-otxt"
    ]

    try:
        subprocess.run(command, check=True)

        if os.path.exists(txt_output):
            with open(txt_output, "r") as f:
                return f.read()
        else:
            return "Transcription failed: output not found."

    except subprocess.CalledProcessError as e:
        return f"Error running whisper.cpp: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    transcription = ""
    if request.method == "POST":
        if "audio_file" in request.files:
            audio = request.files["audio_file"]
            if audio.filename != "":
                path = os.path.join(UPLOAD_FOLDER, audio.filename)
                audio.save(path)

                transcription = transcribe_with_whisper_cpp(path)

    return render_template("index.html", transcription=transcription)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
