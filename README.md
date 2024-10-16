<img src="https://i.postimg.cc/prSMmXy4/lumio-ai.png" alt="Poster" width="400"/>

# Lumio AI Glass - Enhancing Accessibility for the Visually Impaired

## Inspiration
The inspiration for **Lumio AI Glass** was born out of a desire to make a meaningful impact on the lives of visually impaired individuals. We recognized the daily challenges faced by people with visual impairments and aimed to leverage cutting-edge technology to enhance accessibility and independence. We were inspired by the idea of creating a smart wearable device that would act as a reliable companion, providing real-time information and assistance in various aspects of life.

## What It Does
**Lumio AI Glass** is a multifunctional wearable device designed to assist the visually impaired in their daily activities. It combines various technologies to provide a range of features:

- **Object Recognition**: Through deep learning and computer vision, the device can recognize and describe objects in the user's environment. This includes identifying everyday objects and providing audio descriptions.
  
- **Voice Interaction**: Users can interact with the device using voice commands. Lumio AI Glass can recognize and respond to voice prompts, enabling tasks such as object identification and voice-guided navigation.

- **Gesture Control**: We integrated gesture recognition using the MediaPipe library, allowing users to perform specific actions through hand gestures. This touchless interaction method adds an extra layer of convenience.

- **Database Management**: We've implemented a database system using PostgreSQL to securely store relevant data and user information. This will play a crucial role in user customization and data retrieval.

## How We Built It
Building **Lumio AI Glass** was a multidisciplinary effort that involved multiple technologies and domains:

- **Deep Learning**: We adopted the YOLOv8 model for object detection, fine-tuning it for our specific use case. This was crucial for the device's ability to recognize a wide range of objects.

- **Speech Recognition**: To implement voice interaction, we integrated speech recognition using the SpeechRecognition library. This enables users to provide voice commands naturally.

- **Gesture Control**: We utilized the MediaPipe library to recognize hand gestures and translate them into commands for the AI Glass.

- **Database Management**: PostgreSQL is used to store and efficiently manage data, including detecting and storing text, and retrieving matching text from the database.

## Challenges We Ran Into
Our journey in developing **Lumio AI Glass** came with its fair share of challenges:

- **Complex AI Models**: Implementing and fine-tuning complex deep learning models like YOLOv8 required significant effort and expertise.
  
- **Real-Time Interaction**: Achieving real-time interaction and responsiveness for features like object recognition and voice interaction was a technical challenge.

## Accomplishments That We're Proud Of
We're proud of the milestones we've achieved in developing **Lumio AI Glass**:

- Successfully implemented deep learning models for object recognition, enabling users to identify and interact with their surroundings.

- Integrated voice recognition technology for natural voice commands, improving accessibility.

- Developed a gesture control system that enhances the device's usability and user experience.

## What We Learned
Our journey with **Lumio AI Glass** has been a valuable learning experience:

- We gained expertise in deep learning models, including object detection and fine-tuning.

- We mastered the integration of speech recognition and gesture control for an enhanced user interface.

- We honed our teamwork and project management skills, essential for tackling multifaceted projects.

## What's Next for Lumio AI
The journey doesn't end here. We have ambitious plans for the future of **Lumio AI Glass**:

- **Improved Object Recognition**: We aim to enhance the device's object recognition capabilities, making it even more accurate and versatile.

- **Advanced Voice Interaction**: We plan to integrate natural language processing to enable more complex and context-aware voice interactions.

- **Gesture Control Expansion**: Expanding gesture control features to cover a broader range of actions and gestures for user convenience.

- **Connectivity**: We plan to explore connectivity options, such as Bluetooth and Wi-Fi, to enable seamless integration with other smart devices and platforms.

- **Wearable Design**: Focusing on the physical design of the AI Glass to make it practical, comfortable, and stylish for everyday use.

## Built With
- **YOLOv8** for object recognition
- **SpeechRecognition** library for voice interaction
- **MediaPipe** for gesture control
- **PostgreSQL** for database management
- **Python** for development
- **OpenCV** for computer vision
