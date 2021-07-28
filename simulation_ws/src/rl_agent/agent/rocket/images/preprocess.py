from collections import deque
from skimage import transform  # Help us to preprocess the frames
import numpy as np
import cv2


def resize(input_image, shape):
    """resize images
    Args:
      input_image: images data
      shape: target shape to resize the images.
    Returns:
      Resized Image
    """
    if isinstance(shape, tuple):
        shape = np.array(shape)

    input_image = cv2.resize(
        input_image, (shape[0], shape[1]), interpolation=cv2.INTER_NEAREST)
    return input_image


def normalize(input_image, interval):
    """normalize images match the interval
    Args:
      input_image: images data
      interval : [lower, upper]
    Returns:
      Normalized Image
    """
    # Normalize Pixel Values
    if isinstance(interval, tuple):
        interval = np.array(interval)
    assert len(interval) == 2

    # [lower, upper] = interval
    # normalized_frame = input_image / 255.0 * (upper-lower) + lower

    normalized_frame = input_image / 255.0 * \
        (interval[1] - interval[0]) + interval[1]

    return normalized_frame


"""
    preprocess_frame:
    Take a frame.
    Resize it.
        __________________
        |                 |
        |                 |
        |                 |
        |                 |
        |_________________|

        to
        _____________
        |            |
        |            |
        |            |
        |____________|
    Normalize it.

    return preprocessed_frame

    """


def preprocess_frame(frame, interval=(0, 1), shape=[128, 128, 3], out_channel='channel_first', noise=False):
    # x = np.mean(frame,-1)

    if isinstance(shape, tuple):
        shape = np.array(shape)

    # Resize
    preprocessed_frame = resize(frame, shape)

    if out_channel == 'channel_first':
        preprocessed_frame = np.rollaxis(preprocessed_frame, -1, 0)

    # Normalize Frame into [0,1]
    normalized_frame = normalize(preprocessed_frame, interval)

    # Add noise if desired
    return normalized_frame


def stack_frames(stacked_frames, state, is_new_episode, shape=[128, 128, 3], stack_size=4, out_channel='channel_first'):
    if isinstance(shape, tuple):
        shape = np.array(shape)

    # Preprocess frame
    frame = preprocess_frame(state, shape=shape, out_channel=out_channel)

    if out_channel == 'channel_first':
        shape = np.rollaxis(np.array(shape), -1, 0)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros(tuple(shape), dtype=np.float64)
                               for i in range(stack_size)], maxlen=stack_size)

        # Because we're in a new episode, copy the same frame 4x
        for _ in range(stack_size):
            stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=0)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=0)

    if out_channel == 'channel_first':
        stacked_state = np.rollaxis(stacked_state, -3, 0)

    return stacked_state, stacked_frames
