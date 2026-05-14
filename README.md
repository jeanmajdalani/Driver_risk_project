# Hybrid Driver Risk Assessment System

## Description

This project is a real-time computer vision system for driver risk assessment.

It detects:

- Eye closure
- Yawning
- Phone distraction
- Head posture / looking away
- Driver risk level over time

The system combines several models and uses temporal fusion to avoid false alerts from short actions.

The final output is a driver state:

- SAFE
- WARNING
- DANGEROUS

---

## Motivation

Driver drowsiness and distraction can cause serious road accidents.

A driver looking away for one second may not be dangerous, but looking away for several seconds can become risky.  
A normal blink should also not be classified as drowsiness.

For this reason, this project uses temporal logic instead of relying only on frame-by-frame predictions.

---

## Main Features

- Real-time webcam-based detection
- Eye open / closed detection using a trained Keras model
- Yawning detection using facial landmarks and a trained machine learning model
- Phone detection using YOLOv8
- Head posture estimation using facial landmarks
- Temporal fusion risk score
- Visual interface with live risk level
- SAFE / WARNING / DANGEROUS states

---

## System Architecture

```text
Camera Input
     ↓
Face Detection / Facial Landmarks
     ↓
Eye Detection Module
Yawn Detection Module
Phone Detection Module
Head Pose Module
     ↓
Temporal Fusion Risk Score
     ↓
Final Driver State
SAFE / WARNING / DANGEROUS
````

---

## System Modules

### 1. Eye State Detection

Detects whether the driver's eyes are open or closed.

It uses:

* MediaPipe FaceMesh to locate the eye region
* Cropping of the left and right eyes
* A trained Keras model: `best_eye_model_v3.keras`
* Temporal logic to measure how long the eyes remain closed

This helps avoid false alerts caused by normal short blinks.

### 2. Yawning Detection

Detects whether the driver is yawning.

It uses:

* MediaPipe facial landmarks
* Mouth landmark distances
* Mouth aspect ratio features
* A trained machine learning model: `best_yawn_landmark_model.pkl`
* An imputer file: `yawn_imputer.pkl`
* Yawn duration and yawn count for temporal analysis

Yawning increases the risk score, especially when it lasts for several seconds or appears with other risky behaviors.

### 3. Phone Detection

Detects if the driver is using or holding a phone.

It uses:

* YOLOv8 object detection
* The model file: `yolov8n.pt`
* Confidence thresholding
* Temporal smoothing to avoid unstable detection

Phone usage is considered a high-risk distraction.

### 4. Head Posture Detection

Estimates the direction of the driver's head.

It detects whether the driver is looking:

* Forward
* Left
* Right
* Down
* Away

The system measures how long the driver is looking away.
Short head movements are not immediately considered dangerous, while long look-away durations increase the risk.

### 5. Fusion Risk Score

Combines all detection results into one final risk level.

It uses:

* Eye closure duration
* Yawn duration
* Phone usage duration
* Look-away duration
* Weighted risk score
* Hard safety rules

The final state is:

```text
SAFE       → normal driving behavior
WARNING    → risky behavior started or moderate risk detected
DANGEROUS  → dangerous behavior continued for several seconds
```

---

## Why Temporal Fusion?

A simple frame-by-frame system can create many false alarms.

Examples:

* A normal blink should not trigger danger.
* Looking right for 0.5 seconds may be normal.
* Looking away for more than 2 seconds can be dangerous.
* Holding a phone for several seconds should increase the risk.
* Combining yawning with eye closure should increase the danger level.

Therefore, this project uses temporal logic to make the system more realistic for real driving situations.

---

## Repository Structure

```text
Driver_risk_project/
│
├── fusion_risk_score.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebooks/
    ├── fusion_risk_score.ipynb
    ├── head_postur_test.py
    ├── preprocessing_yawn_eye.py
    ├── best_eye_model_v3.keras
    ├── best_yawn_landmark_model.pkl
    ├── yawn_imputer.pkl
    └── yolov8n.pt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/jeanmajdalani/Driver_risk_project.git
cd Driver_risk_project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Requirements

The main libraries used in this project are:

```text
opencv-python
numpy
tensorflow
mediapipe
matplotlib
joblib
ultralytics
scikit-learn
pandas
```

---

## How to Run

Run the main fusion system:

```bash
python fusion_risk_score.py
```

The webcam will open and the real-time interface will display:

* Eye state
* Eye closed duration
* Yawn state
* Yawn probability
* Phone detection
* Head direction
* Look-away duration
* Final risk score
* Final level: SAFE / WARNING / DANGEROUS

Press `Q` or `ESC` to stop the program.

---

## Dataset

This project uses multiple datasets and data sources for different modules.

### Eye Dataset

Used to train the eye open / closed classification model.

Classes:

* Open eyes
* Closed eyes

### Yawn Dataset

Used to train the yawning detection model using mouth and facial landmark features.

Classes:

* Yawning
* Not yawning

### Phone Detection

YOLOv8 is used for phone detection.
The system detects the `cell phone` object class in real time.

### Custom Testing

The complete fusion system was tested using webcam input and recorded videos to simulate real driver behavior.

The datasets are not included directly in the repository because of size limitations.

---

## Models and Weights

The trained models used in this project are stored in the `notebooks/` folder:

```text
best_eye_model_v3.keras
best_yawn_landmark_model.pkl
yawn_imputer.pkl
yolov8n.pt
```

---

## Results and Evaluation

| Module          | Method                       | Output                        |
| --------------- | ---------------------------- | ----------------------------- |
| Eye Detection   | Keras model + eye crop       | Open / Closed                 |
| Yawn Detection  | Landmark features + ML model | Yawning / Not yawning         |
| Phone Detection | YOLOv8                       | Phone detected / Not detected |
| Head Posture    | MediaPipe landmarks          | Forward / Left / Right / Down |
| Fusion System   | Temporal weighted risk score | SAFE / WARNING / DANGEROUS    |

---

## Design Choice Experiment

### Frame-by-frame decision vs Temporal fusion

A frame-by-frame decision system gives alerts immediately when one risky frame appears.
This can create false alerts because normal driving includes short blinks and short head movements.

The temporal fusion system is more stable because it considers the duration of each behavior.

| Behavior                             | Frame-by-frame system | Temporal fusion system                     |
| ------------------------------------ | --------------------- | ------------------------------------------ |
| Normal blink                         | May detect danger     | Ignored                                    |
| Looking right for 0.5 seconds        | May detect warning    | Ignored or low risk                        |
| Looking away for more than 2 seconds | Warning / dangerous   | Dangerous                                  |
| Eyes closed for more than 2 seconds  | Dangerous             | Dangerous                                  |
| Phone visible for several seconds    | Dangerous             | Dangerous                                  |
| Yawning for several seconds          | Warning / dangerous   | Warning or dangerous depending on duration |

This design makes the system more realistic for real driving conditions.

---

## Real-Time Interface

The system displays a live OpenCV interface showing:

```text
Eye State
Eyes Closed Duration
Yawn State
Yawn Probability
Phone Object Detection
Head Direction
Look-Away Duration
Final Risk Score
Final Level
```

The color of the displayed risk level changes according to the final state:

* Green: SAFE
* Yellow: WARNING
* Red: DANGEROUS

---

## Demo Video

Demo video link: (https://drive.google.com/file/d/1QNx58-er9rVoL4yA_m1jmE_AZK7DbbTL/view?usp=sharing)

The demo video shows:

1. Normal driving state
2. Eye closure detection
3. Yawning detection
4. Phone detection
5. Head posture / looking away detection
6. Final risk level changing in real time

---

## Limitations

* The system depends on camera angle and lighting conditions.
* Detection accuracy can decrease if the face is partially hidden.
* Phone detection depends on whether the phone is visible in the camera frame.
* Head pose estimation is based on landmarks and may be affected by extreme angles.
* More testing is needed inside a real car environment.
* Dataset diversity can be improved for better generalization.

---

## Future Work

Future improvements include:

* Add a phone notification alert system.
* Send alerts to the driver's phone or an emergency contact.
* Improve night-time detection.
* Test the system with more real driving videos.
* Optimize the system for higher FPS.
* Deploy the system on embedded hardware such as Raspberry Pi or NVIDIA Jetson.
* Add seatbelt or driver absence detection.

---

## Phone Alert Extension

As a future extension, the system can send an alert to the driver's phone when the risk level remains dangerous for several seconds.

Example:

```text
If final risk = DANGEROUS for more than 3 seconds:
    Send alert to phone
```

Possible implementations:

* Telegram bot notification
* SMS API
* Mobile application notification
* Emergency contact alert

In the current version, the system focuses on real-time visual risk detection and proposes the phone alert system as future work.

---
