import cv2
import numpy as np
import pyautogui
from gaze_tracking import GazeTracking

# Prevent mouse from going off screen
pyautogui.FAILSAFE = True

# Initialize Gaze Tracking
gaze = GazeTracking()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for calibration and smoothing
calibrated_center = None
prev_x, prev_y = 0, 0
alpha = 0.2  # Smoothing factor for mouse movement

def main():
    global calibrated_center, prev_x, prev_y
    screen_w, screen_h = pyautogui.size()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirror-like experience
        frame = cv2.flip(frame, 1)

        # Analyze gaze using GazeTracking
        gaze.refresh(frame)
        frame = gaze.annotated_frame()

        # Display instructions for calibration
        if calibrated_center is None:
            cv2.putText(
                frame, "Look at the center and press 'C' to calibrate",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        else:
            cv2.putText(
                frame, "Gaze tracking active. Press 'Q' to quit.",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

        # Get normalized pupil positions
        pupil_left = gaze.pupil_left_coords()
        pupil_right = gaze.pupil_right_coords()

        if pupil_left and pupil_right:
            # Average pupil position for both eyes
            avg_pupil_x = (pupil_left[0] + pupil_right[0]) / 2
            avg_pupil_y = (pupil_left[1] + pupil_right[1]) / 2

            if calibrated_center is None:
                # Wait for calibration input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    # Calibrate center based on current gaze
                    calibrated_center = (avg_pupil_x, avg_pupil_y)
                    print(f"Calibrated center: {calibrated_center}")
                    continue
            else:
                # Calculate offsets from the calibrated center
                offset_x = avg_pupil_x - calibrated_center[0]
                offset_y = avg_pupil_y - calibrated_center[1]

                # Map offsets to screen coordinates
                x = np.interp(offset_x, [-50, 50], [0, screen_w])
                y = np.interp(offset_y, [-50, 50], [0, screen_h])

                # Smooth mouse movement
                smoothed_x = alpha * x + (1 - alpha) * prev_x
                smoothed_y = alpha * y + (1 - alpha) * prev_y
                prev_x, prev_y = smoothed_x, smoothed_y

                # Move the mouse
                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0.1)

        # Display the frame
        cv2.imshow('Gaze Tracking', frame)

        # Exit on pressing 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
