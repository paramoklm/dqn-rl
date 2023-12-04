import numpy as np

def combine_frames(frame1, frame2):
    # Define your combining function here (example: sum of frames)
    return np.maximum(frame1, frame2)

def fold_frames_with_last(frame_stack, combining_function):
    num_frames, height, width = frame_stack.shape

    folded_frames = []
    for i in range(num_frames - 1):
        combined = combining_function(frame_stack[i], frame_stack[i + 1])
        folded_frames.append(combined)

    # Handling the last frame alone
    folded_frames.append(frame_stack[-1])

    return np.array(folded_frames)

# Example stack of frames (4, 84, 84) as you mentioned
frame_stack = np.random.rand(4, 84, 84)

# Perform the fold operation as specified
folded_result = fold_frames_with_last(frame_stack, combine_frames)
print(folded_result.shape)  # Output the shape of the folded result
