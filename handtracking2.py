import cv2 as cv
import mediapipe as mp
vid = cv.VideoCapture("vid.mp4")
# vid = cv.VideoCapture(0)
mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
hands =mpHands.Hands()#the default values are okay()

while True:
  ret,frame = vid.read()

  if ret ==True:
    frame =cv.resize(frame, (500,500))
    frameBGR = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
    results = hands.process(frameBGR)
    
    #to check for any hands
    if results.multi_hand_landmarks:
      for result in results.multi_hand_landmarks:
        for id_,lm in (enumerate(result.landmark)):
          h,w,c = frame.shape
          cx,cy = int(lm.x*w), int(lm.y*h)
          if id_==0:
            cv.circle(frame,(cx,cy),5,(255,2550),2)
        mpDraw.draw_landmarks(frame,result,mpHands.HAND_CONNECTIONS)
    cv.imshow("Frame",frame)
    if cv.waitKey(2) & 0xFF == ord("d"):
      break
  else:
    print("Somethings  wrong ur with camera")
    break




vid.release()
cv.destroyAllWindows()




# so the hands tracking module has two main module ie hand  landmarks and palm detection
#first create an object for the class hands
#From the mpHands.Hands()
  #static_image_mode=> if false it only detects the hand if the detection confidence is significant
  #the rest are self explanatory

