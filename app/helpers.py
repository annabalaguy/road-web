import base64
import numpy as np


def image_to_dict(image_array, dtype='uint8', encoding='utf-8'):
    '''
    Convert an ndarray representing a batch of images into a compressed string

    ----------
    Parameters
    ----------
    imgArray: a np array representing an image

    ----------
    Returns
    ----------
    dict(image: str,
         height: int,
         width: int,
         channel: int)
    '''
    # Get current shape, only for single image
    if image_array.ndim < 2 or image_array.ndim > 4:
        raise TypeError
    elif image_array.ndim < 3:
        image_array.reshape(*image_array.shape, 1)
    elif image_array.ndim < 4:
        size = 1
        height, width, channel = image_array.shape
    elif image_array.ndim > 4:
        size, height, width, channel = image_array.shape

    # Ensure uint8
    image_array = image_array.astype(dtype)
    # Flatten image
    image_array = image_array.reshape(size * height * width * channel)
    # Encode in b64 for compression
    image_array = base64.b64encode(image_array)
    # Prepare image for POST request, ' cannot be serialized in json
    image_array = image_array.decode(encoding).replace("'", '"')

    api_dict = {'image': image_array, 'size': size, 'height': height,
                'width': width, 'channel': channel}

    return api_dict


def image_from_dict(api_dict, dtype='uint8', encoding='utf-8'):
    '''
    Convert an dict representing a batch of images into a ndarray

    ----------
    Parameters
    ----------
    api_dict: a dict(image, height, width, channel) representing an image
    dtype: target data type for ndarray
    encoding: encoding used for image string

    ----------
    Returns
    ----------
    ndarray of shape (size, height, width, channel)
    '''
    # Decode image string
    img = base64.b64decode(bytes(api_dict.get('image'), encoding))
    # Convert to np.ndarray and ensure dtype
    img = np.frombuffer(img, dtype=dtype)
    # Reshape to original shape
    img = img.reshape((api_dict.get('size'),
                      api_dict.get('height'),
                      api_dict.get('width'),
                      api_dict.get('channel')))

    return img
