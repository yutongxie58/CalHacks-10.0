
# Object Detection
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
# Annotate With Boxes
from supervision import ColorPalette, Color
from supervision import Detections, BoxAnnotator

# Speech Recognition
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import joblib


tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('text_classification_model.pkl')


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
    
        model = YOLO("importedModels/yolov8x.pt")  # load a pretrained YOLOv8x model
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
        
    
    def scan(self, run_duration=3):

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
        if self.prominent != None:
            return self.prominent
        else:
            return object
        


class Data:
    def __init__(self):
        self.output = []
        self.input = []
        self.pastwords = []
        self.dictionary = {"Scan this item": "Scan", "what is this": "Voiceout","read it for me":"Reading","say it out":"Voiceout"}

    def getInput(self, voice):
        self.input.append(voice)
    
    def getOutput(self, image):
        self.output.append(image)

class Act:
    def __init__(self, data):
        self.data = data
    def scan(self):
        detector = ObjectDetection(capture_index=0)
        scanResult = detector.scan(run_duration=3)
        if scanResult:
            self.data.output.append(scanResult)
            text_to_speech(self.data.output[0])
        else:
            text_to_speech("failed to detect object")
    
    
    def voiceout(self):
        if self.data.output==[]:
            text_to_speech("I have nothing in my head right now.")
        else:
            text_to_speech(self.data.output[0])
            self.data.pastwords.append(self.data.output[0])
            self.data.output = self.data.output[1:]

    def controll(self):
        t = self.data.input[0]
        a=tfidf_vectorizer.transform([t])
        c = svm_model.predict(a)[0]
        print(c)
        if c == "voiceout":
            self.voiceout()
        elif c == "scan":
            self.scan()
        elif c == "impolite":
            text_to_speech("and you, my friend, you are the real hero")
        else:
            text_to_speech("I cant Hear You")
        self.data.input = self.data.input[1:]

def text_to_speech(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
    playsound.playsound("output.mp3")
    os.remove("output.mp3")

# Initialize Data and Act objects


def openImageClassifier():
    data = Data()
    act = Act(data)
    # Speech recognition
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            text_to_speech("What can I do for you?")
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
    cv2.destroyAllWindows()