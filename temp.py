print("Starting hand tracking...")
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model ONCE
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Open webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        break

    # Convert BGR -> RGB for Mediapipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process frame
    results = hands.process(rgb_img)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display
    cv2.imshow("Hand Tracking", img)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
