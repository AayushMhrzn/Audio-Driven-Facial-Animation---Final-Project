# AVATAR AI - AUDIO DRIVEN FACIAL ANIMATION 
An AI-powered interactive 3D avatar that predict generates facial animations directly from audio input using deep learning, CNN and TCN model. The primary objective was to design a lightweight and practical framework capable of synthesizing lip motion without requiring video input, motion capture systems, or depth sensors.
Such systems are essential for virtual Avatars, Anchors and assistive communication technologies.
My project demonstrates **speech-driven facial animation**, where an avatar lip-syncs based on input audio (uploaded audio file or live audio input).

---

## 🚀 Features

- 🎤 **Live Voice Input** (Microphone)
- 📂 **Audio file Upload**
- 🎭 **3D Avatar Animation**
---

## 🧠 System Pipeline

`Audio Input → MFCC Extraction → CNN-TCN Model → Lip Landmarks → Blendshapes → 3D Avatar Animation`

## Setup guide

- Requirements.txt is provided to install dependencies.
- change to correct directory to run the script. 
`cd .\scripts\`
   
## Start Backend (Flask)
`python app.py`

## Start Frontend Server
- change directory to web to run the frontend then run the command
`python -m http.server 5500`



