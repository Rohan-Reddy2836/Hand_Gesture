import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
print(dir(vision))
base_options = python.BaseOptions(
    model_asset_path="./hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
landmarker = vision.HandLandmarker.create_from_options(options)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20),
    (0,17)
]
camera = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
print(screen_width, screen_height)
button_color = (255,0,0)
text_color = (255,255,255)
button_x = 50
button_y = 50
button_width = 180
button_height = 80
pyautogui.FAILSAFE = True
previous_x = 0
previous_y = 0
SMOOTHING = 5
pinch_active = False
while True:
    success, frame = camera.read()
    if not success:
        break
    button_color = (255,0,0)
    text_color = (255,255,255)
    button_clicked = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_frame
    )
    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        height, width, _ = frame.shape
        for hand in result.hand_landmarks:
            index_tip = hand[8]
            thumb_tip = hand[4]
            index_x = int(index_tip.x * width)
            index_y = int(index_tip.y * height)
            thumb_x = int(thumb_tip.x * width)
            thumb_y = int(thumb_tip.y * height)
            screen_x = int(index_x * screen_width / width)
            screen_y = int(index_y * screen_height / height)
            current_x = previous_x + (screen_x - previous_x) / SMOOTHING
            current_y = previous_y + (screen_y - previous_y) / SMOOTHING
            pyautogui.moveTo(current_x, current_y)
            previous_x = current_x
            previous_y = current_y
            for landmark in hand:
                pixel_x = int(landmark.x * width)
                pixel_y = int(landmark.y * height)
                cv2.circle(
                frame,
                (pixel_x, pixel_y),
                5,
                (0,255,0),
                -1
                )
            for start, end in HAND_CONNECTIONS:
                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)
                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)
                cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
                )
            cv2.circle(
            frame,
            (index_x, index_y),
            12,
            (0,0,255),
            -1
            )
            cv2.circle(
            frame,
            (thumb_x, thumb_y),
            12,
            (255,0,255),
            -1
            )
            distance = math.hypot(
            thumb_x - index_x,
            thumb_y - index_y
            )
            cv2.putText(
            frame,
            f"Distance: {int(distance)}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,255),
            2   
            )
            cv2.putText(
            frame,
            f"({index_x}, {index_y})",
            (index_x + 15, index_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
            )
            hovering = (
            button_x < index_x < button_x + button_width and
            button_y < index_y < button_y + button_height
            )
            if hovering:
                    button_color = (0,255,0)
                
                    if hovering and distance < 40:
                        if not pinch_active:
                            pinch_active = True
                            pyautogui.click()
                            button_clicked = True
                            print("CLICK!")
                    else:
                        pinch_active = False
                    if button_clicked:
                        cv2.putText(
                            frame,
                            "BUTTON CLICKED!",
                            (150,100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3
                            )
    cv2.rectangle(
    frame,
    (button_x, button_y),
    (button_x + button_width, button_y + button_height),
    button_color,
    3
    )
    cv2.putText(
    frame,
    "BUTTON",
    (button_x + 25, button_y + 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    text_color,
    2
    )
    cv2.imshow("GB-HCI Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break   
camera.release()
cv2.destroyAllWindows()


