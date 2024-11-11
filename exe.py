import cv2
import numpy as np
import tensorflow as tf 

cap = cv2.VideoCapture(0)                     # activate camera
model = tf.keras.models.load_model('mnist.h5')   # load model

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, img = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(img,(540,300))          # change the size of image to fasten the efficiency
    x, y, w, h = 400, 200, 60, 60            # Define the position and size of the region for extracting numbers.
    img_num = img.copy()                     # Duplicate an image for recognition use.
    img_num = img_num[y:y+h, x:x+w]          # Capture the region for recognition.

    img_num = cv2.cvtColor(img_num, cv2.COLOR_BGR2GRAY)   
    # Convert color to grayscale.
    ret, img_num = cv2.threshold(img_num, 127, 255, cv2.THRESH_BINARY_INV)   
    # Perform binarization for text, converting it to black-and-white with white text on a black background.
    output = cv2.cvtColor(img_num, cv2.COLOR_GRAY2BGR)     
    # convert to BGR
    img[0:60, 480:540] = output                           
    # Display the converted image in the top-right corner of the screen.


    img_num = cv2.resize(img_num,(28,28)) # Resize to 28x28 and compare with the trained model.
    img_num = img_num.astype(np.float32)  
    img_num = np.array(img_num)  
    img_num = img_num.reshape(-1,28,28,1)

   
    img_num = img_num/255
    img_pre = model.predict(img_num)          # recognize
    img_pre = list(img_pre)
    img_pre = list(img_pre[0])

    num = str(img_pre.index(max(img_pre)))      # get the result
    text = num                              # ext content.
    org = (x,y-20)                          # text position.
    fontFace = cv2.FONT_HERSHEY_SIMPLEX     # text font
    fontScale = 2                           # text size
    color = (0,0,255)                       # text color
    thickness = 2                           # thickness of the text border.
    lineType = cv2.LINE_AA                  # style of the text border.
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType) 
    # print it out


    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)  # mark the recognition area.
    cv2.imshow('oxxo', img)
    if cv2.waitKey(50) == 27:
        break                            # press esc to stop
cap.release()
cv2.destroyAllWindows()