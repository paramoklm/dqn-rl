import cv2  
import numpy as np

def preprocess_frame(frame, prev_frame=None):
    '''
    Function to preprocess a single frame (with previous frame)
    '''
    if prev_frame is not None:
        # Take the maximum value for each pixel color value over the current frame and the previous frame
        frame = np.maximum(frame, prev_frame)

    yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

    # Extract the Y channel (luminance)
    y_channel = yuv_image[:,:,0]

    y_channel_scale = cv2.resize(y_channel, (84, 84), interpolation=cv2.INTER_AREA)

    return y_channel_scale

    # Extract the Y channel (luminance) from the RGB frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Rescale the frame to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

    return frame

def stack_frames(frames, visualize=False):
    '''
    Function to apply preprocessing to the m most recent frames and stack them
    '''
    # Stack the frames along the depth dimension to produce the input to the Q-function
    input = np.stack(frames, axis=2)

    if visualize:
        return input

    return input.transpose(2, 0, 1)  # Change shape from (84, 84, 4) to (4, 84, 84)
    # return input[np.newaxis, :] # Shape to (1, 4, 84, 84)



# def preprocess(frames, m, visualize=False):
#     '''
#     Preprocessing to the m most recent frames
#         The last frame in frames is the most recent one (the current one)
#     '''
#     preprocessed_frames = []
#     m = len(frames)
#     for i in range(m - 1):
#         # Process frames two by two
#         preprocessed_frames.append(preprocess_frame(frames[m - 1 - i], frames[m - 2 - i]))
#     # Process last frame alone
#     preprocessed_frames.append(preprocess_frame(frames[0]))
# 
#     return stack_frames(preprocessed_frames, visualize=visualize)


def combine_frames(frame1, frame2):
    # Define your combining function here (example: sum of frames)
    return np.maximum(frame1, frame2)

def preprocess(frame_stack):
    num_frames, height, width = frame_stack.shape

    folded_frames = []
    for i in range(num_frames - 1):
        combined = combine_frames(frame_stack[i], frame_stack[i + 1])
        folded_frames.append(combined)

    # Handling the last frame alone
    folded_frames.append(frame_stack[-1])

    return np.stack(folded_frames).astype(np.uint8)