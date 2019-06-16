import ctypes as ct
import logging

import numpy as np

from pyflif.flif_wrapper_common import FlifImageBase, FlifDecoderBase

__all__ = ["FlifDecoderImage", "FlifDecoder"]

Logger = logging.getLogger("FLIF_Decoder")
Logger.setLevel("WARN")


class FlifDecoderImage(FlifImageBase):
    flif_image_handle = None

    @property
    def width(self):
        return self.Flif.get_width(self.flif_image_handle)

    @property
    def height(self):
        return self.Flif.get_height(self.flif_image_handle)

    @property
    def nb_channels(self):
        return self.Flif.get_nb_channels(self.flif_image_handle)

    @property
    def depth(self):
        return self.Flif.get_depth(self.flif_image_handle)

    @property
    def palette_size(self):
        return self.Flif.get_palette_size(self.flif_image_handle)

    row_reader = None
    dtype = None
    mImgShape = None

    def __init__(self, flif_image_handle):
        self.flif_image_handle = flif_image_handle

        if 0 != self.palette_size:
            raise ValueError("")

        dtype = (np.uint8, np.uint16)

        if 1 == self.nb_channels:
            row_reader = (self.Flif.read_row_GRAY8, self.Flif.read_row_GRAY16)
            self.mImgShape = (self.height, self.width)
        else:
            row_reader = (self.Flif.read_row_RGBA8, self.Flif.read_row_RGBA16)
            self.mImgShape = (self.height, self.width, 4)

        if 8 == self.depth:
            self.dtype = dtype[0]
            row_reader = row_reader[0]
        elif 16 == self.depth:
            self.dtype = dtype[1]
            row_reader = row_reader[1]
        else:
            assert False  # depth should be always 8 or 16

        self.row_reader = lambda row_idx, buffer, buf_size: \
            row_reader(self.flif_image_handle, row_idx, buffer, buf_size)

    def get_image(self):
        aux_shape = (self.mImgShape[0], np.prod(self.mImgShape[1:]))

        npy_img = np.zeros(aux_shape, dtype=self.dtype)
        assert npy_img.flags['C_CONTIGUOUS']

        img_pointer = npy_img.ctypes.data_as(ct.c_void_p)

        for row_idx in range(npy_img.shape[0]):
            self.row_reader(row_idx, img_pointer, npy_img.strides[0])
            img_pointer.value += npy_img.strides[0]  # jump to next row (strides in bytes)

        npy_img = npy_img.reshape(self.mImgShape)

        if (1 < self.nb_channels) and (4 != self.nb_channels):
            npy_img = npy_img[:, :, :self.nb_channels]

        return npy_img


class FlifDecoder(FlifDecoderBase):
    # Object members
    flif_decoder_handle = None
    mFname = None

    def __init__(self, fname):
        self.mFname = fname
        self.flif_decoder_handle = None

    def __enter__(self):
        # create decoder
        self.flif_decoder_handle = self.Flif.create_decoder()
        Logger.debug("Create FLIF decoder %r", self.flif_decoder_handle)

        # set CRC check
        self.Flif.set_crc_check(self.flif_decoder_handle, 1)

        # decode file
        if 0 == self.Flif.decode_file(self.flif_decoder_handle, self.mFname.encode('utf-8')):
            raise IOError("Error decoding FLIF file %r" % self.mFname)

        Logger.debug("Decoded FLIF file %r", self.mFname)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.flif_decoder_handle is not None:
            self.Flif.destroy_decoder(self.flif_decoder_handle)
            self.flif_decoder_handle = None

    def num_images(self):
        if self.flif_decoder_handle is None:
            raise IOError("FLIF file %r not (yet) decoded" % self.mFname)
        return self.Flif.num_images(self.flif_decoder_handle)

    def get_image(self, index):
        if index >= self.num_images():
            raise ValueError("Frame index %d out of %d requested" % (index, self.num_images()))

        # optain image pointer
        flif_image_handle = self.Flif.get_image(self.flif_decoder_handle, index)

        if flif_image_handle is None:
            raise IOError("Error reading image %d" % index)

        return FlifDecoderImage(flif_image_handle).get_image()
