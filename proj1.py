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

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
svm_model = joblib.load('text_classification_model.pkl')


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        self.box_annotator = BoxAnnotator(
            color=ColorPalette(colors=[Color(r=255, g=0, b=0), Color(r=0, g=255, b=0), Color(r=0, g=0, b=255)]),
            thickness=3, text_thickness=3, text_scale=1.5)

        self.prominent = None

    def load_model(self):

        model = YOLO("yolov8l.pt")  # load a pretrained YOLOv8x model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):

        # Filter out "person" class results
        filtered_results = [result for result in results[0] if
                            self.CLASS_NAMES_DICT[int(result.boxes.cls.cpu().numpy().astype(int))] != "person"]
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
        filtered_results = [result for result in results[0] if
                            self.CLASS_NAMES_DICT[int(result.boxes.cls[0].cpu().numpy().astype(int))] != "person"]
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

            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, 'Scan', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            # f'FPS: {int(fps)}'
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
        self.pastwords = []
        self.dictionary = {"check it out": "Scan", "what is this": "Voiceout", "read it for me": "Reading",
                           "say it out": "Voiceout"}

    def getInput(self, voice):
        self.input.append(voice)

    def getOutput(self, image):
        self.output.append(image)


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
        Textimage.video_capture.release()
        Textimage.cv2.destroyAllWindows()
        self.data.getOutput(Textimage.result)

    def voiceout(self):
        if self.data.output == []:
            text_to_speech("I have nothing in my head right now.")
        else:
            text_to_speech(self.data.output[0])
            self.data.pastwords.append(self.data.output[0])
            self.data.output = self.data.output[1:]

    def controll(self):
        t = self.data.input[0]
        a = tfidf_vectorizer.transform([t])
        c = svm_model.predict(a)[0]
        print(c)
        if c == "voiceout":
            self.voiceout()
        elif c == "scan":
            self.scan()
        elif c == "impolite":
            text_to_speech("and you, my friend, you are the real hero")
        else:
            text_to_speech("I can't help you with that")
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

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# Function to connect to CockroachDB and compare extracted text
class Textimage:
    result = ""

    def compare_to_database(words):
        # print("words:",words)
        results = []

        import os
        for word in words:
            try:
                # Connect to CockroachDB
                conn = psycopg2.connect(os.environ["DATABASE_URL"]
                                        )

                cursor = conn.cursor()
                cursor.execute(f"SELECT product_name FROM product WHERE LOWER(product_name) = LOWER('{word}')")

                result = cursor.fetchone()

                cursor.close()
                conn.close()

                if result: results.append(result)

            except Exception as e:
                print("Error:", e)

        return results

    # Preprocessing
    # Open webcam
    video_capture = cv2.VideoCapture(0)

    def image_threshold(image):
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    # removing noise
    def remove_noise(image, kernel_size=3):
        result = cv2.medianBlur(image, ksize=kernel_size)
        return result

    # dilation
    def dilation(image):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(image, kernel, iterations=1)
        return result

    # closing
    def closing(image):
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return result

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        threshold = image_threshold(gray)
        noise_removed = remove_noise(threshold)
        # dilated = dilation(noise_removed)
        # processed_image = closing(noise_removed)

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
        db_result = compare_to_database(words)

        if db_result:
            result = f"Match found in database: {db_result[0]}"
            quit()
            # Perform the necessary action when a match is found

        # print(repr(extracted_text))

        # boxes
        boxes = pytesseract.image_to_data(noise_removed, output_type=pytesseract.Output.DICT)

        for i in range(len(boxes['text'])):
            if int(boxes['conf'][i]) > 0.5:  # Adjust confidence threshold as needed
                (x, y, w, h) = (boxes['left'][i], boxes['top'][i], boxes['width'][i], boxes['height'][i])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(frame, boxes['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # cv2.putText(frame, extracted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Video', frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(1)

    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'