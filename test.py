import cv2 as cv
import mediapipe as mp
vid = cv.VideoCapture("vid.mp4")
# vid = cv.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands =mpHands.Hands()#the default values are okay()



while True:
  let,frame = vid.read()
  if let:
    frame = cv.resize(frame, (500,500))
    frameBGR = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    results = hands.process(frameBGR)
    if results.multi_hand_landmarks:
      for hlm in results.multi_hand_landmarks:
        for id_,lm in enumerate(hlm.landmark):
          h,w,c = frame.shape
          cx,cy = int(lm.x*w),int(lm.y*h)
          if id_ == 0:
            cv.circle(frame, (cx,cy),5, (250,233,233),20)
        mpDraw.draw_landmarks(frame,hlm,mpHands.HAND_CONNECTIONS)
    cv.imshow("frame",frame)
    if cv.waitKey(2) & 0xFF == ord("d"):
      break
  else:
    break

vid.release()
cv.destroyAllWindows()