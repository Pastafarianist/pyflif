import ctypes as ct
import logging

import numpy as np

from pyflif.flif_wrapper_common import FlifImageBase, FlifEncoderBase

__all__ = ["FlifEncoderImage", "FlifEncoder"]

Logger = logging.getLogger("FLIF_Encoder")
Logger.setLevel("WARN")


class FlifEncoderImage(FlifImageBase):
    mImage = None
    mImporter = None

    flif_image_handle = None

    def __init__(self, np_image):
        self.mImporter = self.get_flif_importer(np_image)
        self.mImage = self.correct_image_strides(np_image)

    def __enter__(self):
        assert (0 == (self.mImage.strides[0] % self.mImage.ndim))
        self.flif_image_handle = self.mImporter(self.mImage.shape[1], self.mImage.shape[0],
                                                self.mImage.ctypes.data_as(ct.c_void_p),
                                                self.mImage.strides[0] / self.mImage.ndim)

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
        if 2 == np_image.ndim:
            # gray scale image
            img_type = "gray"
            importer = (flif.import_image_GRAY, flif.import_image_GRAY16)
        elif (3 == np_image.ndim) and (3 == np_image.shape[2]):
            # RGB Image
            img_type = "RGB"
            importer = (flif.import_image_RGB,)
        elif (3 == np_image.ndim) and (4 == np_image.shape[2]):
            # RGBA Image
            img_type = "RGBA"
            importer = (flif.import_image_RGBA,)
        else:
            raise ValueError("Unsupported image shape '%s'" % repr(np_image.shape))

        # check dtype
        if np.issubdtype(np_image.dtype, np.uint8):
            importer = importer[0]
        elif np.issubdtype(np_image.dtype, np.uint16) and (2 == len(importer)):
            importer = importer[1]
        else:
            raise TypeError(f"image dtype {np_image.dtype:!r} in combination with {img_type} not supported")

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

    mDoCrcCheck = None
    mInterlaced = None
    mLearnRepeat = None
    mSplitThreshold = None
    mMaxLoss = None

    def __init__(self, fname, crc_check=True, interlaced=False, learn_repeat=4, split_threshold_factor=12, maxloss=0):
        self.fname = fname

        self.mDoCrcCheck = int(bool(crc_check))
        self.mInterlaced = int(bool(interlaced))
        self.mLearnRepeat = max(0, int(learn_repeat))
        self.mSplitThreshold = 5461 * 8 * max(4, int(split_threshold_factor))
        self.mMaxLoss = max(0, min(100, int(maxloss)))

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

        self.Flif.set_interlaced(self.flif_encoder_handle, self.mInterlaced)
        self.Flif.set_learn_repeat(self.flif_encoder_handle, self.mLearnRepeat)
        self.Flif.set_split_threshold(self.flif_encoder_handle, self.mSplitThreshold)
        self.Flif.set_crc_check(self.flif_encoder_handle, self.mDoCrcCheck)
        self.Flif.set_lossy(self.flif_encoder_handle, self.mMaxLoss)

        return self

    def close(self):
        if self.flif_encoder_handle is not None:
            try:
                retval = self.Flif.encode_file(self.flif_encoder_handle, self.fname)
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
