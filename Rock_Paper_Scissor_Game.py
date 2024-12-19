import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os

# Initialize MediaPipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Labels for gestures
GESTURES = ["paper", "rock", "scissors"]

# Load Images for Animation
script_directory = r"C:\\Users\\vishw\\OneDrive\\Desktop\\Coding\\Mini Project"
os.chdir(script_directory)
ai_images = {}
player_images = {}
for key in GESTURES:
    image_path = f"./assets/{key.lower()}.png"  # Path based on gesture name
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Image for {key} not found at {image_path}")
        continue  # Skip loading this image
    ai_images[key] = cv2.resize(img, (200, 200))  # Adjust size as needed
    player_images[key] = cv2.resize(img, (200, 200))  # Player uses the same images

# Check if all images are loaded
if len(ai_images) < len(GESTURES):
    print("Some images are missing. Ensure all images (Rock, Paper, Scissors) are in the 'assets' folder.")
    exit(1)  # Exit the program to avoid further errors

# Initialize Scoreboard
player_score = 0
ai_score = 0

# Simulate AI Move
def get_ai_move():
    return random.choice(GESTURES)

# Determine the winner
def determine_winner(player, ai):
    if player == ai:
        return "Tie"
    elif (player == "rock" and ai == "scissors") or \
         (player == "paper" and ai == "rock") or \
         (player == "scissors" and ai == "paper"):
        return "Player"
    else:
        return "AI"

# Predict gesture based on landmarks (placeholder logic for now)
def predict_gesture(hand_landmarks):
    # Placeholder gesture recognition (Random for now)
    return random.choice(GESTURES)

# Overlay AI move animation
def display_ai_move(frame, ai_move):
    if ai_move in ai_images:
        x_offset, y_offset = frame.shape[1] - 250, 50  # Adjust position on the right
        overlay_image = ai_images[ai_move]
        overlay_h, overlay_w, _ = overlay_image.shape
        frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = overlay_image
    return frame

# Overlay Player move animation
def display_player_move(frame, player_move):
    if player_move in player_images:
        x_offset, y_offset = 50, 50  # Adjust position on the left
        overlay_image = player_images[player_move]
        overlay_h, overlay_w, _ = overlay_image.shape
        frame[y_offset:y_offset+overlay_h, x_offset:x_offset+overlay_w] = overlay_image
    return frame

# Main function to run the game
def play_game_with_camera():
    global player_score, ai_score

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    player_move = "Unknown"
    ai_move = "Waiting"
    winner = ""
    is_player_turn = True
    last_turn_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Handle player's turn
        if is_player_turn:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detect player's move
                    player_move = predict_gesture(hand_landmarks)
                    is_player_turn = False  # End player's turn
                    last_turn_time = time.time()  # Mark the turn timestamp
                    break  # Process only one hand

        # Handle AI's turn after a brief delay
        elif not is_player_turn and time.time() - last_turn_time > 1:  # 1-second delay
            ai_move = get_ai_move()
            winner = determine_winner(player_move, ai_move)

            # Update scores
            if winner == "Player":
                player_score += 1
            elif winner == "AI":
                ai_score += 1

            # Reset for the next round
            is_player_turn = True
            player_move = "Unknown"
            ai_move = "Waiting"

        # Display animations
        if player_move in GESTURES:
            frame = display_player_move(frame, player_move)
        if ai_move in GESTURES:
            frame = display_ai_move(frame, ai_move)

        # Display moves and results
        turn_info = "Player Turn" if is_player_turn else "AI Turn"
        cv2.putText(frame, turn_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Winner: {winner}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Score - Player: {player_score} AI: {ai_score}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Rock Paper Scissors", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game_with_camera()
