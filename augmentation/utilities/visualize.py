import numpy as np


def gallery(array, ncols=None):
    # https://stackoverflow.com/questions/42040747/more-idiomatic-way-to-display-images-in-a-grid-with-numpy
    nindex, height, width, intensity = array.shape
    if ncols is None:
        ncols = int(np.floor(np.sqrt(nindex)))
    while nindex % ncols != 0: ncols += 1

    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1, 2)
              .reshape(height*nrows, width*ncols, intensity))
    return result
