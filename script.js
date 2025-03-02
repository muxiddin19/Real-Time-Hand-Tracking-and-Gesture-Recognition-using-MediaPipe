const video = document.getElementById('video');

// Set up the webcam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            console.log("Webcam stream loaded.");
            resolve(video);
        };
    });
}

// Load the model and detect faces
async function loadAndDetectFaces() {
    const model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
    );
    console.log("Model loaded.");
    await setupCamera();
    video.play();

    const detectFaces = async () => {
        const predictions = await model.estimateFaces({
            input: video,
            returnTensors: false,
            flipHorizontal: false
        });
        console.log("Predictions:", predictions);
        requestAnimationFrame(detectFaces);
    };

    detectFaces();
}

loadAndDetectFaces();
const video = document.getElementById('video');

// Set up the webcam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

// Load the model and detect faces
async function loadAndDetectFaces() {
    const model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
    );
    await setupCamera();
    video.play();

    const detectFaces = async () => {
        const predictions = await model.estimateFaces({
            input: video,
            returnTensors: false,
            flipHorizontal: false
        });
        console.log(predictions);
        requestAnimationFrame(detectFaces);
    };

    detectFaces();
}

loadAndDetectFaces();

