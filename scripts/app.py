import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pydub import AudioSegment
import io
from inference import generate_animation_from_audio


# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

app = Flask(__name__)
CORS(app)


# -----------------------------
# AUDIO PROCESS ENDPOINT
# -----------------------------
@app.route("/process-audio", methods=["POST"])
def process_audio():

    try:

        # Check if audio file exists
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]

        # Read bytes
        audio_bytes = audio_file.read()

        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty audio"}), 400

        print("Audio received:", len(audio_bytes), "bytes")
        # Convert any format → wav
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_bytes = wav_buffer.getvalue()
        # -----------------------------
        # Generate Animation
        # -----------------------------
        animation_data = generate_animation_from_audio(wav_bytes)

        # Encode audio so frontend can play same audio
        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        return jsonify({
            "audio": audio_base64,
            "animation": animation_data
        })


    except Exception as e:

        print("SERVER ERROR:", str(e))

        return jsonify({
            "error": "Processing failed",
            "audio": "",
            "animation": {}
        }), 500


# -----------------------------
# HEALTH CHECK (optional)
# -----------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "Avatar LipSync API Running"
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)