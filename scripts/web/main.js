import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

let scene, camera, renderer, controls;
let model, animationData = null, audio = null;
let fps = 30, clock = new THREE.Clock();
let isSpeaking = false;

let mediaRecorder;
let audioChunks = [];

init();
loadModel();
animate();

/* ---------- THREE INIT ---------- */

function init() {

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x14001f);

    const avatarSection = document.getElementById("avatar-section");

    camera = new THREE.PerspectiveCamera(
        38,
        avatarSection.clientWidth / avatarSection.clientHeight,
        0.1,
        1000
    );

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(avatarSection.clientWidth, avatarSection.clientHeight);
    avatarSection.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = false;

    scene.add(new THREE.AmbientLight(0x673ab7, 1.5));

    const light = new THREE.DirectionalLight(0x7c4dff, 3);
    light.position.set(3, 4, 5);
    scene.add(light);

    window.addEventListener("resize", onWindowResize);
}

/* ---------- LOAD MODEL ---------- */

function loadModel() {

    const loader = new GLTFLoader();

    loader.load("head.gltf", (gltf) => {

        model = gltf.scene;
        scene.add(model);

        model.traverse((child) => {

            if (child.isMesh) {

                const mat = child.material;

                if (child.name.includes("head")) {
                    mat.color.setHex(0xffffff);
                    mat.roughness = 0.5;
                    mat.metalness = 0.2;
                    mat.envMapIntensity = 1.5;
                }

                mat.needsUpdate = true;
            }
        });

        const box = new THREE.Box3().setFromObject(model);
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());

        // Center model
        model.position.sub(center);

        camera.position.set(0, size.y * 0.1, size.z * 1.8);
        controls.target.set(0, size.y * 0.1, 0);

        controls.update();
    });
}

/* ---------- SEND AUDIO TO BACKEND ---------- */

async function sendAudioToServer(blob) {

    const formData = new FormData();
    formData.append("audio", blob, "speech.wav");

    try {

        const response = await fetch("http://localhost:5000/process-audio", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!data.animation) {
            console.error("Invalid response from server");
            return;
        }

        playAudio(data);

    } catch (err) {

        console.error("Server error:", err);
    }
}

/* ---------- PLAY AUDIO + ANIMATION ---------- */

function playAudio(data) {

    animationData = data.animation;
    fps = animationData.fps;

    const audioSrc = "data:audio/wav;base64," + data.audio;
    audio = new Audio(audioSrc);

    isSpeaking = true;

    audio.onended = () => {

        isSpeaking = false;
        animationData = null;
        audio = null;
    };

    audio.play();
}

/* ---------- MICROPHONE RECORDING ---------- */

const micBtn = document.getElementById("micBtn");

micBtn.onclick = async () => {

    if (!mediaRecorder) {

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = e => {
            audioChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {

            const blob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];

            sendAudioToServer(blob);
        };
    }

    if (mediaRecorder.state === "recording") {

        mediaRecorder.stop();
        micBtn.classList.remove("listening");

    } else {

        audioChunks = [];
        mediaRecorder.start();
        micBtn.classList.add("listening");
    }
};

/* ---------- AUDIO UPLOAD ---------- */

const uploadBtn = document.getElementById("uploadBtn");

uploadBtn.onchange = function () {

    const file = this.files[0];

    if (!file) return;

    sendAudioToServer(file);
};

/* ---------- ANIMATION LOOP ---------- */

function animate() {

    requestAnimationFrame(animate);

    controls.update();

    // Idle movement
    if (model && !isSpeaking) {

        const t = clock.getElapsedTime();

        model.rotation.y = Math.sin(t * 0.5) * 0.08;
        model.rotation.x = Math.sin(t * 0.3) * 0.03;
    }

    // Lip sync animation
    if (animationData && audio && model) {

        const frameIndex = Math.floor(audio.currentTime * fps);

        if (frameIndex < animationData.frames.length) {

            const frame = animationData.frames[frameIndex];

            model.traverse((child) => {

                if (child.morphTargetDictionary) {

                    animationData.blendshapes.forEach(name => {

                        const i = child.morphTargetDictionary[name];

                        if (i !== undefined) {
                            child.morphTargetInfluences[i] = frame[name];
                        }
                    });
                }
            });
        }
    }

    renderer.render(scene, camera);
}

/* ---------- RESIZE ---------- */

function onWindowResize() {

    const avatarSection = document.getElementById("avatar-section");

    camera.aspect = avatarSection.clientWidth / avatarSection.clientHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(
        avatarSection.clientWidth,
        avatarSection.clientHeight
    );
}