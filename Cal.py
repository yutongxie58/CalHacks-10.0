import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
from supervision import ColorPalette, Color
from supervision import Detections, BoxAnnotator
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import joblib
import pytesseract
from PIL import Image
import psycopg2
import cv2
import mediapipe as mp
import pyttsx3
import subprocess
from pydub import AudioSegment
from pydub.playback import play
import threading

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

tfidf_vectorizer = joblib.load('/Users/jenny/Documents/Machine_Learning/CAL/tfidf_vectorizer-new.pkl')
svm_model = joblib.load('/Users/jenny/Documents/Machine_Learning/CAL/text_classification_model-new.pkl')

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")


class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        self.prominent = None
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = BoxAnnotator(color=ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), Color(r=0, g=0, b=255)]), thickness=3, text_thickness=3, text_scale=1.5)
    

    def load_model(self):
    
        model = YOLO("yolov8l.pt")  # load a pretrained YOLOv8x model
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame)
        
        return results
    

    def plot_bboxes(self, results, frame):
        
        # Filter out "person" class results
        filtered_results = [result for result in results[0] if self.CLASS_NAMES_DICT[int(result.boxes.cls.cpu().numpy().astype(int))] != "person"]
        if not filtered_results:
        # No non-person objects detected
            return frame
        # Extract data for visualization
        xyxys = [result.boxes.xyxy.cpu().numpy() for result in filtered_results]
        confidences = [result.boxes.conf.cpu().numpy() for result in filtered_results]
        class_ids = [result.boxes.cls.cpu().numpy().astype(int) for result in filtered_results]      
        # Concatenate xyxys, confidences, and class_ids if not empty
        # Concatenate xyxys, confidences, and class_ids
        xyxy = np.concatenate(xyxys, axis=0)
        confidence = np.concatenate(confidences, axis=0)
        class_id = np.concatenate(class_ids, axis=0)

        # Setup detections for visualization
        detections = Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
              for confidence, class_id
              in zip(detections.confidence, detections.class_id)]
        
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        return frame
    
    def mostProminent(self, results):
        filtered_results = [result for result in results[0] if self.CLASS_NAMES_DICT[int(result.boxes.cls[0].cpu().numpy().astype(int))] != "person"]
        if filtered_results:
            most_prominent_class_id = filtered_results[0].boxes.cls[0].cpu().numpy().astype(int)
            most_prominent_label = self.CLASS_NAMES_DICT[int(most_prominent_class_id)]
        else:
            most_prominent_label = "No objects detected"
        
        return most_prominent_label
        
    
    def __call__(self, run_duration=3):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        

        start_time = time.time()
        end_time = start_time + run_duration
      
        while time.time() < end_time:
          
            start_time = time.time()
            
            ret, frame = cap.read()
            assert ret
            
            results = self.predict(frame)
            
            frame = self.plot_bboxes(results, frame)

            object = self.mostProminent(results)
            if (object != "No objects detected"):
                self.prominent = object

            fps = 1/np.round(end_time - start_time, 2)
             
            cv2.putText(frame, 'Scan', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #f'FPS: {int(fps)}'
            cv2.imshow('YOLOv8 Detection', frame)
 
            if cv2.waitKey(5) & 0xFF == 27:
                # cv2.destroyWindow('YOLOv8 Detection')
                break  
        cap.release()
        cv2.destroyAllWindows()

class Data:
    def __init__(self):
        self.output = []
        self.input = []

    def getInput(self, voice):
        self.input.append(voice)
  
    def getOutput(self, image):
        self.output.append(image)


# Initialize Data and Act objects
def tooclose():
    cap = cv2.VideoCapture(0)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame_width = int(cap.get(3))  # Capture frame width
    frame_height = int(cap.get(4))  # Capture frame height
    total_frame_area = frame_width * frame_height  # Total frame area


    desired_coverage = 0.4
    some_size_threshold = int(total_frame_area * desired_coverage)

    while True:
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        movement_threshold = 5.5

        moving_pixels = np.where(mag > movement_threshold)

        if moving_pixels[0].any() and moving_pixels[1].any():
            contours, _ = cv2.findContours((mag > movement_threshold).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find the contour with the largest area
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box of the largest moving object

                # Check the size of the largest object as an indicator of proximity
                if w * h > some_size_threshold:
                    text_to_speech("TOO CLOSE watch out")
                    quit()

        cv2.imshow('Moving Object Detection', frame2)

        prvs = next

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




# Function to connect to CockroachDB and compare extracted text
class Textimage:
    def __init__(self) -> None:
        self.video_capture = cv2.VideoCapture(0)

    @staticmethod
    def compare_to_database(words):
        # print("words:",words)
        results = []

        import os
        for word in words:
            try:
                # Connect to CockroachDB
                conn = psycopg2.connect("postgresql://teamcalhacks:zCLkx3Q9lwkOmkA_b3m-mQ@db-ocr-recognition-3703.g95.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full")
                cursor = conn.cursor()
                cursor.execute(f"SELECT product_name FROM product WHERE LOWER(product_name) = LOWER('{word}')")

                result = cursor.fetchone()

                cursor.close()
                conn.close()

                if result: results.append(result)

            except Exception as e:
                print("Error:", e)

        return results

    @staticmethod
    def image_threshold(image):
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    # removing noise
    @staticmethod
    def remove_noise(image, kernel_size=3):
        result = cv2.medianBlur(image, ksize=kernel_size)
        return result
        
    # dilation
    @staticmethod
    def dilation(image):
        kernel= np.ones((5,5), np.uint8)
        result=cv2.dilate(image, kernel, iterations=1)
        return result
        
    # closing
    @staticmethod
    def closing(image):
        kernel = np.ones((5,5), np.uint8)
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result

    def read(self):
        while True:
            ret, frame = self.video_capture.read()

            if not ret:
                break

            # grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            threshold = Textimage.image_threshold(gray)
            noise_removed = Textimage.remove_noise(threshold)
            #dilated = dilation(noise_removed)
            #processed_image = closing(noise_removed)

            # extract text
            extracted_text = pytesseract.image_to_string(noise_removed)

            lines = extracted_text.splitlines()
            words = []

            # Process each line separately
            for line in lines:
                words.extend(line.split())

                # for word in words:
                #     print(word)

            # Compare extracted text to database
            db_result = Textimage.compare_to_database(words)

            if db_result:
                result= f"Match found in database: {db_result[0]}"
                return db_result[0][0]
                # Perform the necessary action when a match is found

            # print(repr(extracted_text))

            # boxes
            boxes = pytesseract.image_to_data(noise_removed, output_type=pytesseract.Output.DICT)

            for i in range(len(boxes['text'])):
                if int(boxes['conf'][i]) > 0.5:  # Adjust confidence threshold as needed
                    (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, boxes['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)



            #cv2.putText(frame, extracted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('Video', frame)

            # Exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(1)



def volume_control():
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

 

class Act:
    def __init__(self, data):
        self.data = data

    def scan(self):
        detector = ObjectDetection(capture_index=0)
        detector(run_duration=3)
        if detector.prominent != None:
            self.data.output.append(detector.prominent)
            text_to_speech(self.data.output[0])
        else:
            text_to_speech("failed to detect object")
        
    def read(self):
        text_image = Textimage()
        
        
        self.data.getOutput(text_image.read())
        self.voiceout()
    
    def voiceout(self):
        if self.data.output==[]:
            text_to_speech("I have nothing in my head right now.")
        else:
            print("debug 1:", self.data.output[0])
            text_to_speech(self.data.output[0])
            self.data.output = self.data.output[1:]

    def controll(self):
        t = self.data.input[0]
        a=tfidf_vectorizer.transform([t])
        c = svm_model.predict(a)[0]
        if c == "scan" or t== "check it out":
            self.scan()
        elif c == "read" or t== "read it for me":
            self.read()
        elif c == "voicecontrol" or t== "voice control":
            volume_control()
        elif c == "becareful" or t== "too close":
            tooclose()
        elif c == "impolite":
            text_to_speech("don't be like that, you are better than this")
        else:
            text_to_speech("I can't help you with that")
        self.data.input = self.data.input[1:]


def openImageClassifier():
    data = Data()
    act = Act(data)
    # Speech recognition
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            text_to_speech("Say something:")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = recognizer.listen(source, timeout=None, phrase_time_limit=3)

        text = recognizer.recognize_google(audio)  # Use Google Web Speech API
        data.getInput(text)
        act.controll()
        # time.sleep(3)
    except sr.UnknownValueError:
        text_to_speech("Sorry, I couldn't understand what you said.")
    except sr.RequestError:
        text_to_speech("I'm having trouble accessing the Google Web Speech API.")

openImageClassifier()