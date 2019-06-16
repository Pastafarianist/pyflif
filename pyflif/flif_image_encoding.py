import ctypes as ct
import logging

import numpy as np

from pyflif.flif_wrapper_common import FlifImageBase, FlifEncoderBase

__all__ = ["FlifEncoderImage", "FlifEncoder"]

Logger = logging.getLogger("FLIF_Encoder")
Logger.setLevel("WARN")


class FlifEncoderImage(FlifImageBase):
    image = None
    importer = None

    flif_image_handle = None

    def __init__(self, np_image):
        self.importer = self.get_flif_importer(np_image)
        self.image = self.correct_image_strides(np_image)

    def __enter__(self):
        assert (0 == (self.image.strides[0] % self.image.ndim))
        self.flif_image_handle = self.importer(
            self.image.shape[1], self.image.shape[0],
            self.image.ctypes.data_as(ct.c_void_p),
            self.image.strides[0] // self.image.ndim,
        )

        Logger.debug("Using FLIF image importer %s", repr(self.flif_image_handle))

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.flif_image_handle is not None:
            self.Flif.destroy_image(self.flif_image_handle)
            self.flif_image_handle = None

    @classmethod
    def get_flif_importer(cls, np_image):
        flif = cls.Flif

        # check images planes
        if np_image.ndim == 2:
            # gray scale image
            img_type = "gray"
            importer = (flif.import_image_GRAY, flif.import_image_GRAY16)
        elif np_image.ndim == 3 and np_image.shape[2] == 3:
            # RGB Image
            img_type = "RGB"
            importer = (flif.import_image_RGB,)
        elif np_image.ndim == 3 and np_image.shape[2] == 4:
            # RGBA Image
            img_type = "RGBA"
            importer = (flif.import_image_RGBA,)
        else:
            raise ValueError("Unsupported image shape {!r}".format(np_image.shape))

        # check dtype
        if np.issubdtype(np_image.dtype, np.uint8):
            importer = importer[0]
        elif np.issubdtype(np_image.dtype, np.uint16) and (2 == len(importer)):
            importer = importer[1]
        else:
            raise TypeError("image dtype {:!r} in combination with {} not supported".format(np_image.dtype, img_type))

        Logger.info("Importing %s%d image", img_type, np_image.itemsize << 3)

        return importer

    @staticmethod
    def correct_image_strides(np_image):

        def is_copy_required(np_image):
            if np_image.strides[-1] != np_image.itemsize:
                return True

            if 3 == np_image.ndim:
                if np_image.strides[1] != (np_image.itemsize * np_image.shape[2]):
                    return True

            return False

        if is_copy_required(np_image):
            Logger.info("Deep copy on image required")
            np_image = np_image.copy(order='C')

        assert not is_copy_required(np_image)

        return np_image


####################################################################################


class FlifEncoder(FlifEncoderBase):
    # Object members
    flif_encoder_handle = None
    fname = None

    do_crc_check = None
    interlaced = None
    learn_repeat = None
    split_threshold = None
    max_loss = None

    def __init__(self, fname, crc_check=True, interlaced=False, learn_repeat=4, split_threshold_factor=12, maxloss=0):
        self.fname = fname

        self.do_crc_check = int(bool(crc_check))
        self.interlaced = int(bool(interlaced))
        self.learn_repeat = max(0, int(learn_repeat))
        self.split_threshold = 5461 * 8 * max(4, int(split_threshold_factor))
        self.max_loss = max(0, min(100, int(maxloss)))

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.destroy()

    def open(self):
        # check if file is writable
        with open(self.fname, "w") as _:
            pass

        self.flif_encoder_handle = self.Flif.create_encoder()
        Logger.debug("Create FLIF encoder %r", self.flif_encoder_handle)

        self.Flif.set_interlaced(self.flif_encoder_handle, self.interlaced)
        self.Flif.set_learn_repeat(self.flif_encoder_handle, self.learn_repeat)
        self.Flif.set_split_threshold(self.flif_encoder_handle, self.split_threshold)
        self.Flif.set_crc_check(self.flif_encoder_handle, self.do_crc_check)
        self.Flif.set_lossy(self.flif_encoder_handle, self.max_loss)

        return self

    def close(self):
        if self.flif_encoder_handle is not None:
            try:
                retval = self.Flif.encode_file(self.flif_encoder_handle, self.fname.encode('utf-8'))
                if 1 != retval:
                    raise IOError("Error writing FLIF file %s" % self.fname)
            finally:
                self.destroy()

    def destroy(self):
        if self.flif_encoder_handle is not None:
            handle = self.flif_encoder_handle
            self.flif_encoder_handle = None
            self.Flif.destroy_encoder(handle)

    def add_image(self, img):
        if self.flif_encoder_handle is not None:
            if isinstance(img, FlifEncoderImage):
                if img.flif_image_handle is not None:
                    self.Flif.add_image(self.flif_encoder_handle, img.flif_image_handle)
            else:
                with FlifEncoderImage(img) as img:
                    self.move_image(img)

    def move_image(self, flif_image):
        assert isinstance(flif_image, FlifEncoderImage), "%r is not a FlifEncoderImage" % flif_image

        if (self.flif_encoder_handle is not None) and (flif_image.flif_image_handle is not None):
            self.Flif.add_image_move(self.flif_encoder_handle, flif_image.flif_image_handle)
