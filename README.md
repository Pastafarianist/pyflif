# `pyflif`: `ctypes`-based Python wrapper for Free Lossless Image Format

## Build Instructions

### Install the dependencies

#### FLIF dependencies

`pyflif` dynamically loads the [FLIF](https://github.com/FLIF-hub/FLIF) library. Please ensure that you have it installed, e.g.:

```bash
$ ldconfig -p | grep flif
    libflif_dec.so.0 (libc6,x86-64) => /usr/lib/libflif_dec.so.0
    libflif.so.0 (libc6,x86-64) => /usr/lib/libflif.so.0
```

#### `pyflif` dependencies

 - numpy: `sudo apt-get install python-numpy` (on debian/ubuntu)
 - scipy (optional): `sudo apt-get install python-scipy` (on debian/ubuntu)
 - opencv (optional): `sudo apt-get install python-opencv` (on debian/ubuntu)

#### Installation

```bash
pip install --user git+https://github.com/Pastafarianist/pyflif.git
```

## Usage

### Decoding

#### Simple method for single images

```python
import pyflif

# img: numpy array with shape [ WxHx(3 or 4) ]
# dtype in (uint8, uint16)
img = pyflif.read_flif("path/to/image.flif")
```

#### Advanced method also for animations

```python
import pyflif

with pyflif.FlifDecoder("path/to/file.flif") as dec:
    # decompress all frames and store them in a list
    allframes = [dec.get_image(idx) for idx in range(dec.num_images())]
```

### Encoding

WARNING: [encoding does not work yet](https://github.com/Pastafarianist/pyflif/issues/1).

#### Simple method for single images

```python
import pyflif

# img: numpy array with shape [ WxHx(3 or 4) ]
# dtype in (uint8, uint16)
pyflif.write_flif("path/to/image.flif", img)
```

#### Advanced method also for animations

```python
import pyflif

with pyflif.FlifEncoder("path/to/file.flif") as enc:
    for img in <image source>:
	    # all images should have the same shape [ WxHx(3 or 4) ] and dtype (uint8/uint16)
	    enc.add_image(img)
```
