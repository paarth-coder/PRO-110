# for capturing frames from webcam
import cv2

# for processing images
import numpy as np

# for testing sample data based on our model
import tensorflow as tf

# attaching webcam indexed as 0 with the application software
camera = cv2.VideoCapture(0)

# making an object/variable named 'mymodel'
mymodel = tf.keras.models.load_model('stone paper scissor.h5')

# Run this loop till camera is opened or connected with the applcation software
while camera.isOpened():

    # requesting a frame from camera
    status , frame = camera.read()

    # if we were able to capture the frame successfully, status is 'true'
    if status:

        # flipping the frame
        frame = cv2.flip(frame , 1)

        # resizing the frame
        resized_frame = cv2.resize(frame , (224 , 224))

        # increasing the dimension of the frame to 4D along axis 0
        resized_frame = np.expand_dims(resized_frame , axis = 0)

        # normalizing the frame (element value ranges from 0 : 255)
        resized_frame = resized_frame / 255

        # getting predictions from the model
        predictions = mymodel.predict(resized_frame)

        # accessing the predictions array to get the percentages
        stone_percent = int(predictions[0][0]*100)
        paper_percent = int(predictions[0][1]*100)
        scissor_percent = int(predictions[0][2]*100)

        # comparing the data to get the prediction with max percentage
        if stone_percent >= paper_percent and stone_percent >= scissor_percent:
            print(f"I am {stone_percent} sure, that this is a stone")
        elif paper_percent >= stone_percent and paper_percent >= scissor_percent:
            print(f"I am {paper_percent} sure, that this is a paper")
        elif scissor_percent >= stone_percent and scissor_percent >= paper_percent:
            print(f"I am {scissor_percent} sure, that this is a scissor")

        # displaying the video feed
        cv2.imshow('video feed' , frame)

        # waiting for key press for 1ms
        code = cv2.waitKey(1)

        # if 'b' key is pressed, break
        if code  ==  ord('b'):
            break

camera.release()
cv2.destroyAllWindows()
