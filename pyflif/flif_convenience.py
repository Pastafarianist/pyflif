import os.path

from pyflif.flif_image_decoding import FlifDecoder
from pyflif.flif_image_encoding import FlifEncoder

try:
    import scipy.misc

    generic_img_reader = lambda path: scipy.misc.imread(path)
    generic_img_writer = lambda path, img: scipy.misc.imsave(path, img)
except ImportError:
    try:
        import cv2

        generic_img_reader = lambda path: cv2.imread(path)
        generic_img_writer = lambda path, img: cv2.imwrite(path, img)
    except ImportError:
        generic_img_reader = None
        generic_img_writer = None

__all__ = ["imwrite", "imread"]


def imwrite(path, img):
    ext = os.path.splitext(path)[1]

    if ".flif" == ext.lower():
        with FlifEncoder(path) as enc:
            return enc.add_image(img)
    elif generic_img_reader is not None:
        return generic_img_writer(path, img)
    else:
        raise IOError("%r is not a FLIF file" % path)


def imread(path):
    ext = os.path.splitext(path)[1]

    if ".flif" == ext.lower():
        with FlifDecoder(path) as dec:
            return dec.get_image(0)
    elif generic_img_reader is not None:
        return generic_img_reader(path)
    else:
        raise IOError("%r is not a FLIF file" % path)
