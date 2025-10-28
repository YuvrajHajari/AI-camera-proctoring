/* THIS IS THE CORRECTED IMPORT
  We are importing the entire module as an object named 'cocoSsd'
*/
// THIS IS THE CORRECT, EXPLICIT PATH TO THE ES MODULE FILE
import * as cocoSsd from "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@latest/dist/coco-ssd.esm.js";
// Import the MediaPipe components we already had
import {
    FaceLandmarker,
    FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";


// Run our main function once the document is fully loaded
window.addEventListener("DOMContentLoaded", main);

async function main() {
    // Get references to our HTML elements
    const video = document.getElementById('webcam');
    const faceCountDisplay = document.getElementById('face-count');
    const objectFlagDisplay = document.getElementById('object-flag');

    // A variable to hold our loaded models
    let faceLandmarker;
    let objectDetector;

    try {
        // --- STEP 1: LOAD THE MODELS (BOTH OF THEM) ---
        faceCountDisplay.textContent = "Loading models...";

        // Load FaceLandmarker (as before)
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU" 
            },
            runningMode: "VIDEO",
            numFaces: 5 
        });
        console.log("FaceLandmarker model loaded successfully!");

        /* THIS IS THE CORRECTED FUNCTION CALL
          We call the .load() function on the 'cocoSsd' object we imported
        */
        objectDetector = await cocoSsd.load();
        console.log("Object detection (coco-ssd) model loaded successfully!");


        // --- STEP 2: ENABLE THE WEBCAM ---
        faceCountDisplay.textContent = "Waiting for webcam...";
        
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
            throw new Error("Webcam access (getUserMedia) is not supported by this browser.");
        }
        
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        video.addEventListener("loadeddata", () => {
            console.log("Webcam ready, starting prediction loop.");
            predictWebcam(faceLandmarker, objectDetector, video, faceCountDisplay, objectFlagDisplay);
        });

    } catch (error) {
        console.error("Failed to initialize:", error);
        faceCountDisplay.style.backgroundColor = "red"; 
        faceCountDisplay.textContent = `ERROR: ${error.message}`;
    }
}

/**
 * The main detection loop.
 */
let lastVideoTime = -1; 

async function predictWebcam(faceLandmarker, objectDetector, video, faceCountDisplay, objectFlagDisplay) {
    const startTimeMs = performance.now();

    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;

        // --- RUN BOTH MODELS ---
        const faceResults = faceLandmarker.detectForVideo(video, startTimeMs);
        const objectResults = await objectDetector.detect(video);

        
        // --- PROCESS FACE RESULTS ---
        const faceCount = faceResults.faceLandmarks.length;
        if (faceCount === 0) {
            faceCountDisplay.style.backgroundColor = "rgba(255, 100, 100, 0.7)"; 
            faceCountDisplay.textContent = "FLAG: No Face Detected!";
        } else if (faceCount > 1) {
            faceCountDisplay.style.backgroundColor = "rgba(255, 100, 100, 0.7)"; 
            faceCountDisplay.textContent = "FLAG: Multiple Faces Detected!";
        } else {
            faceCountDisplay.style.backgroundColor = "rgba(0, 0, 0, 0.7)";
            faceCountDisplay.textContent = `Faces Detected: ${faceCount}`;
        }

        // --- PROCESS OBJECT RESULTS ---
        let objectDetected = false;
        for (const object of objectResults) {
            if (object.class === 'cell phone' || object.class === 'book') {
                objectDetected = true;
                break; 
            }
        }

        if (objectDetected) {
            objectFlagDisplay.textContent = "FLAG: Phone or Book Detected!";
            objectFlagDisplay.style.display = 'block'; 
        } else {
            objectFlagDisplay.style.display = 'none'; 
        }
    }

    window.requestAnimationFrame(() => 
        predictWebcam(faceLandmarker, objectDetector, video, faceCountDisplay, objectFlagDisplay)
    );
}