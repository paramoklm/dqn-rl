import cv2  
import numpy as np

def preprocess_frame(frame, prev_frame=None):
    '''
    Function to preprocess a single frame (with previous frame)
    '''
    if prev_frame is not None:
        # Take the maximum value for each pixel color value over the current frame and the previous frame
        frame = np.maximum(frame, prev_frame)

    # Extract the Y channel (luminance) from the RGB frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Rescale the frame to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    return frame

def stack_frames(frames, m=4):
    '''
    Function to apply preprocessing to the m most recent frames and stack them
    '''
    # Stack the frames along the depth dimension to produce the input to the Q-function
    return np.stack(frames, axis=2)


def preprocess(frames, m):
    '''
    Preprocessing to the m most recent frames
        The last frame in frames is the most recent one (the current one)
    '''
    preprocessed_frames = []
    for i in range(m - 1):
        # Process frames two by two
        preprocessed_frames.append(preprocess_frame(frames[m - 1 - i], frames[m - 2 - i]))
    # Process last frame alone
    preprocessed_frames.append(preprocess_frame(frames[0]))
    
    return stack_frames(preprocessed_frames)