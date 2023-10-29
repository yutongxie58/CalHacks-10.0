import cv2
import pytesseract
import numpy as np
from PIL import Image
import psycopg2
import time

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'


# Function to connect to CockroachDB and compare extracted text
def compare_to_database(words):
    # print("words:",words)
    results = []

    import os
    for word in words:
        try:
            # Connect to CockroachDB
            conn = psycopg2.connect( os.environ["DATABASE_URL"]
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
    kernel= np.ones((5,5), np.uint8)
    result=cv2.dilate(image, kernel, iterations=1)
    return result
    
# closing
def closing(image):
    kernel = np.ones((5,5), np.uint8)
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
    db_result = compare_to_database(words)

    if db_result:
        print(f"Match found in database: {db_result[0]}")
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



    #cv2.putText(frame, extracted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Video', frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1)

video_capture.release()
cv2.destroyAllWindows()


