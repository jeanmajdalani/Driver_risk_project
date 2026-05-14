# Hybrid Driver Risk Assessment System

## Description
This project is a real-time computer vision system for driver risk assessment.

It detects:
- Eye closure
- Yawning
- Phone distraction
- Head posture
- Driver risk level over time

The system combines several models and uses temporal fusion to avoid false alerts from short actions.

## Motivation
Driver drowsiness and distraction can cause serious accidents.  
The goal of this system is to detect risky behavior early and generate warnings.

## System Modules

1. Eye State Detection  
Detects if the driver's eyes are open or closed.

2. Yawning Detection  
Uses facial landmarks to detect yawning.

3. Phone Detection  
Uses YOLO to detect phone usage.

4. Head Posture Detection  
Detects if the driver is looking forward, left, right, down, or away.

5. Fusion Risk Score  
Combines all modules over time to classify the driver state as safe, warning, or danger.

## Repository Structure

```text
Driver_risk_project/
│
├── fusion_risk_score.py
├── requirements.txt
├── README.md
├── notebooks/
│   ├── fusion_risk_score.ipynb
│   ├── head_postur_test.py
│   ├── preprocessing_yawn_eye.py
│   ├── best_eye_model_v3.keras
│   ├── best_yawn_landmark_model.pkl
│   ├── yawn_imputer.pkl
│   └── yolov8n.pt