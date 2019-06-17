import ctypes as ct
import logging
from ctypes.util import find_library

__all__ = ["FlifImageBase", "FlifEncoderBase", "FlifDecoderBase"]

Logger = logging.getLogger("FLIF_wrapper_common")
Logger.setLevel("WARN")


def config_call_general(libflif, struct, flif_prefix, name, argtypes=None, restype=None):
    setattr(struct, name, libflif.__getitem__("{}_{}".format(flif_prefix, name)))
    if argtypes is not None:
        struct.__dict__[name].argtypes = argtypes
    struct.__dict__[name].restype = restype


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

        struct = cls.Flif

        struct.destroy_image = libflif.flif_destroy_image
        struct.destroy_image.argtypes = [ct.c_void_p]

        # Image import function
        img_importers = ["RGBA", "RGB", "GRAY", "GRAY16"]
        #                        width        height       data   major-stride
        import_arg_types = [ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_uint32]

        for importer in img_importers:
            config_call_general(libflif, struct, "flif", "import_image_%s" % importer, import_arg_types, ct.c_void_p)

        # Getter functions
        img_getter = ["width", "height", "nb_channels", "depth", "palette_size"]
        getter_res_type = [ct.c_uint32, ct.c_uint32, ct.c_uint8, ct.c_uint8, ct.c_uint32]

        for getter, rtype in zip(img_getter, getter_res_type):
            config_call_general(libflif, struct, "flif_image", "get_%s" % getter, [ct.c_void_p], rtype)

        # Row Reader
        row_reader = ["GRAY8", "GRAY16", "RGBA8", "RGBA16"]
        reader_args = [ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_size_t]

        for reader in row_reader:
            config_call_general(libflif, struct, "flif_image", "read_row_%s" % reader, reader_args)


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
        Logger.debug("Initializing FlifEncoder")

        struct = cls.Flif

        struct.create_encoder = libflif.flif_create_encoder
        struct.create_encoder.restype = ct.c_void_p

        struct.destroy_encoder = libflif.flif_destroy_encoder
        struct.destroy_encoder.restype = None
        struct.destroy_encoder.argtypes = [ct.c_void_p]

        def config_call(name, argtypes=None, restype=None):
            config_call_general(libflif, struct, "flif_encoder", name, argtypes, restype)

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
        Logger.debug("Initializing FlifDecoder")

        struct = cls.Flif

        struct.create_decoder = libflif.flif_create_decoder
        struct.create_decoder.restype = ct.c_void_p

        struct.destroy_decoder = libflif.flif_destroy_decoder
        struct.destroy_decoder.restype = None
        struct.destroy_decoder.argtypes = [ct.c_void_p]

        def config_call(name, argtypes=None, restype=None):
            config_call_general(libflif, struct, "flif_decoder", name, argtypes, restype)

        config_call("decode_file", [ct.c_void_p, ct.c_char_p], ct.c_int32)
        config_call("set_crc_check", [ct.c_void_p, ct.c_uint32])
        config_call("num_images", [ct.c_void_p], ct.c_size_t)
        config_call("get_image", [ct.c_void_p, ct.c_size_t], ct.c_void_p)


####################################################################################
# Loading DLL or shared library file


def _load_libflif():
    libflif_name = find_library('flif')
    Logger.debug("Loading FLIF library from {}".format(libflif_name))

    libflif = ct.cdll[libflif_name]

    FlifEncoderBase.initialize(libflif)
    FlifImageBase.initialize(libflif)
    FlifDecoderBase.initialize(libflif)


_load_libflif()
