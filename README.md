# AVATAR AI - AUDIO DRIVEN FACIAL ANIMATION 

This is my Final Project for Bachelor's Degree in Computer Engineering, which takes Audio Input (Uploaded Audio file or Mic Input) and then generates lip movement and animates 3D AVATAR based on the output predicted by the CNN-TCN model.

The primary objective was to design a lightweight and practical framework capable of synthesizing lip motion without requiring Video input, motion capture systems. 

---

## 🚀 Features

- 🎤 **Live Voice Input** (Microphone)
- 📂 **Audio file Upload**
- 🎭 **3D Avatar Animation**
---

## 🧠 System Pipeline

`Audio Input → MFCC Extraction → CNN-TCN Model → Lip Landmarks → Blendshapes → 3D Avatar Animation`

---

## Setup guide

- Requirements.txt is provided to install dependencies.
- change to correct directory to run the script. 
`cd .\scripts\`
   
### Start Backend (Flask)
`python app.py`

### Start Frontend Server
- change directory to web to run the frontend then run the command
`python -m http.server 5500`

---

## 🎬 Project Demo

- Open http://localhost:5500 to view the project in the browser.
- Click on `Upload Audio` to upload audio file from computer.
- Choose file to upload wav or mp3 audio containing speech.

https://github.com/user-attachments/assets/cf1e5636-382a-42cf-9528-2ab84ce7737a

---

## AUTHOR

© 2026 Aayush Maharjan. All rights reserved.
This project is developed for academic and demonstration purposes.  
