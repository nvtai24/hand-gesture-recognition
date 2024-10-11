import cv2
import mediapipe as mp


# Initialize MediaPipe Hands solution and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

# Variable to store the previous state of fingers
prev_fingers = [0, 0, 0, 0, 0]

# Initialize the Hands model with detection and tracking confidence
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            break

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB as MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB frame to detect hand landmarks
        result = hands.process(rgb_frame)

        # Default finger state (no hand detected)
        current_fingers = [0, 0, 0, 0, 0]  # Default all to 0 (down)

        # Check if any hands are detected
        if result.multi_hand_landmarks:
            # Loop through all detected hands
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get the hand label: 'Left' or 'Right'
                hand_label = result.multi_handedness[idx].classification[0].label

                # Thumb: check its horizontal movement (left or right) based on hand type
                if hand_label == "Right":
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < \
                       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                        current_fingers[0] = 1  # Thumb is up
                else:
                    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > \
                       hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                        current_fingers[0] = 1  # Thumb is up

                # Check other fingers: index, middle, ring, pinky
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < \
                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
                    current_fingers[1] = 1  # Index finger is up

                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < \
                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
                    current_fingers[2] = 1  # Middle finger is up

                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < \
                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y:
                    current_fingers[3] = 1  # Ring finger is up

                if hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < \
                   hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y:
                    current_fingers[4] = 1  # Pinky finger is up

                # Draw hand landmarks and connections on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Only process and display if the state has changed
        if current_fingers != prev_fingers:
            print(f"Fingers changed: {current_fingers}")
            prev_fingers = current_fingers.copy()  # Update previous state

        # Show the processed frame
        cv2.imshow("Hand Tracking", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
