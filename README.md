# NatyaNetra 
> Real-time Bharatanatyam Mudra recognition using MediaPipe and a trained MLP neural network (99.72% accuracy)

**NatyaNetra** captures hand gestures via webcam, extracts 21 3D landmarks using MediaPipe, and classifies them into Bharatanatyam mudras using a lightweight Neural Network trained on the Bharatanatyam Mudra Dataset.

## Features
- Real-time hand landmark detection via MediaPipe
- MLP classifier with 99.72% validation accuracy
- Supports 6 classical mudras: Pathaka, Tripathaka, Mushti, Ardhachandran, Hamsasyam, Alapadmam
- Live confidence-based gesture stabilisation
- Python-based scoring system
- Optional MSPM0 microcontroller integration via serial

## Tech Stack
`Python` `MediaPipe` `PyTorch` `OpenCV` `NumPy`

## Dataset
[Bharatanatyam Mudra Dataset](https://www.kaggle.com/datasets/onkarpathrikar/bharatanatyam-mudra-dataset-master-bm) by Onkar Pathrikar on Kaggle
