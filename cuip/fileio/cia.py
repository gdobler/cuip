import numpy as np

class CuipImageArray(np.ndarray):
    """
    Parameters
    ----------
    input_array: np.ndarray
        stacked numpy array. example
        an 'rgb' image array of 2160 x 4096 resolution
        with 10 stacked images
        will have shape of shape = (10, 2160, 4096, 3)
    metadata: dict, optional
        optional information about the file in a dict
    comment: optional
        any python datatype. preferably a string.
    """

    def __new__(cls, img_array, comment=None, metadata=None ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(img_array).view(cls)
        # add the new attribute to the created instance
        obj.comment = comment
        obj.metadata = metadata
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.comment = getattr(obj, 'comment', None)
        self.metadata = getattr(obj, 'metadata', None)
