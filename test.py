import mediapipe as mp
import cv2 as cv

cap = cv.VideoCapture("vid.mp4")
mpHands= mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands = mpHands.Hands()


while True:
  let,frame = cap.read()
  frame = cv.resize(frame, (500,500))
  frameBGR = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
  result = hands.process(frameBGR)
  if result.multi_hand_landmarks:
    for hlm in result.multi_hand_landmarks:
      mpDraw.draw_landmarks(frame,hlm,mpHands.HAND_CONNECTIONS)  
  cv.imshow("Frame",frame)
  if cv.waitKey(10) & 0xFF == ord("d"):
    break

cap.release()
cv.destroyAllWindows()
