import ctypes as ct
import logging
from ctypes.util import find_library
from pathlib import Path

import numpy as np

__all__ = ["FlifImageBase", "FlifEncoderBase", "FlifDecoderBase"]

Logger = logging.getLogger("FLIF_wrapper_common")
Logger.setLevel("WARN")

libflif_dir = Path('/usr/lib')
libflif_name = find_library('flif')


class FlifImageBase(object):
    class Flif(object):
        get_width = None
        get_height = None
        get_nb_channels = None
        get_depth = None
        get_palette_size = None

        import_image_RGBA = None
        import_image_RGB = None
        import_image_GRAY = None
        import_image_GRAY16 = None

        read_row_GRAY8 = None
        read_row_GRAY16 = None
        read_row_RGBA8 = None
        read_row_RGBA16 = None

        destroy_image = None

    @classmethod
    def initialize(cls, libflif):
        Logger.debug("Initializing FlifImageBase")

        strct = cls.Flif

        strct.destroy_image = libflif.flif_destroy_image
        strct.destroy_image.argtypes = [ct.c_void_p]

        def config_call_general(flif_prefix, name, argtypes=None, restype=None):
            setattr(strct, name,
                    libflif.__getitem__("%s_%s" % (flif_prefix, name)))
            if argtypes is not None:
                strct.__dict__[name].argtypes = argtypes
            strct.__dict__[name].restype = restype

        # Image import function
        img_importers = ["RGBA", "RGB", "GRAY", "GRAY16"]
        #                        width        height       data   major-stride
        import_arg_types = [ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_uint32]
        config_import_call = lambda name: config_call_general("flif", name, import_arg_types, ct.c_void_p)

        for importer in img_importers:
            config_import_call("import_image_%s" % importer)

        # Getter functions
        img_getter = ["width", "height", "nb_channels", "depth", "palette_size"]
        getter_res_type = [ct.c_uint32, ct.c_uint32, ct.c_uint8, ct.c_uint8, ct.c_uint32]
        config_getter_call = lambda name, rtype: config_call_general("flif_image", name, [ct.c_void_p], rtype)

        for getter, rtype in zip(img_getter, getter_res_type):
            config_getter_call("get_%s" % getter, rtype)

        # Row Reader
        row_reader = ["GRAY8", "GRAY16", "RGBA8", "RGBA16"]
        reader_args = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_size_t]
        config_reader_call = lambda name: config_call_general("flif_image", name, reader_args)

        for reader in row_reader:
            config_reader_call("read_row_%s" % reader)


class FlifEncoderBase(object):
    class Flif(object):
        create_encoder = None
        destroy_encoder = None

        set_interlaced = None
        set_learn_repeat = None
        set_split_threshold = None
        set_crc_check = None
        set_lossy = None

        encode_file = None

        add_image = None
        add_image_move = None

    @classmethod
    def initialize(cls, libflif):
        Logger.debug("Initializing flifEncoder")

        strct = cls.Flif

        strct.create_encoder = libflif.flif_create_encoder
        strct.create_encoder.restype = ct.c_void_p

        strct.destroy_encoder = libflif.flif_destroy_encoder
        strct.destroy_encoder.restype = None
        strct.destroy_encoder.argtypes = [ct.c_void_p]

        def config_call(name, argtypes=None, restype=None):
            setattr(strct, name,
                    libflif.__getitem__("flif_encoder_%s" % name))
            if argtypes is not None:
                strct.__dict__[name].argtypes = argtypes
            strct.__dict__[name].restype = restype

        config_call("encode_file", [ct.c_void_p, ct.c_char_p], ct.c_int32)
        config_call("add_image", [ct.c_void_p, ct.c_void_p])
        config_call("add_image_move", [ct.c_void_p, ct.c_void_p])

        config_call("set_interlaced", [ct.c_void_p, ct.c_uint32])
        config_call("set_learn_repeat", [ct.c_void_p, ct.c_uint32])
        config_call("set_split_threshold", [ct.c_void_p, ct.c_int32])
        config_call("set_crc_check", [ct.c_void_p, ct.c_uint32])
        config_call("set_lossy", [ct.c_void_p, ct.c_int32])


class FlifDecoderBase(object):
    class Flif(object):
        create_decoder = None
        decode_file = None
        set_crc_check = None

        num_images = None
        get_image = None

        destroy_decoder = None

    @classmethod
    def initialize(cls, libflif):
        Logger.debug("Initializing flifDecoder")

        strct = cls.Flif

        strct.create_decoder = libflif.flif_create_decoder
        strct.create_decoder.restype = ct.c_void_p

        strct.destroy_decoder = libflif.flif_destroy_decoder
        strct.destroy_decoder.restype = None
        strct.destroy_decoder.argtypes = [ct.c_void_p]

        def config_call(name, argtypes=None, restype=None):
            setattr(strct, name,
                    libflif.__getitem__("flif_decoder_%s" % name))
            if argtypes is not None:
                strct.__dict__[name].argtypes = argtypes
            strct.__dict__[name].restype = restype

        config_call("decode_file", [ct.c_void_p, ct.c_char_p], ct.c_int32)
        config_call("set_crc_check", [ct.c_void_p, ct.c_uint32])
        config_call("num_images", [ct.c_void_p], ct.c_size_t)
        config_call("get_image", [ct.c_void_p, ct.c_size_t], ct.c_void_p)


####################################################################################
# Loading DLL or shared library file


def _load_libflif():
    Logger.debug(f"Loading FLIF library from {libflif_dir / libflif_name}")
    libflif = np.ctypeslib.load_library(libflif_name, str(libflif_dir))

    FlifEncoderBase.initialize(libflif)
    FlifImageBase.initialize(libflif)
    FlifDecoderBase.initialize(libflif)


_load_libflif()
