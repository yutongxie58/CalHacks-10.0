import cv2
import mediapipe as mp
import pyttsx3
import subprocess
from pydub import AudioSegment
from pydub.playback import play
import threading

# sound file for volume level
ding_sound = AudioSegment.from_wav("/Users/jenny/Downloads/ding.wav")

def play_ding_sound():
    ding_sound.export("ding.wav", format="wav")
    subprocess.run(["afplay", "ding.wav"]) 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# hand gesture
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,max_num_hands = 1,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")

    h, w, c = image.shape
    framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:                
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for hand_landmarks in results.multi_hand_landmarks:
            
            pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            pinky_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
            for hand_lm in hand_landmarks.landmark:
                x, y = int(hand_lm.x * w), int(hand_lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if thumb_tip_y < thumb_mcp_y:
            subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 5)"])
            play(ding_sound)
            threading.Thread(target=play_ding_sound).start()
            
        if thumb_tip_y > thumb_mcp_y:
            subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 5)"])
            play(ding_sound)
            threading.Thread(target=play_ding_sound).start()
   
    cv2.imshow('Detection Window', image)
  
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()




