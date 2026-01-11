# User Manual

## Requirements

The library requires the following:

*   Python 3.8+
*   Numpy

## 1. Single WCS Instance

The library provides specific classes for each projection type to handle transformations between Sky Coordinates (RA, Dec) and Pixel Coordinates (x, y).

Supported projections:
*   `Tan` (Gnomonic)
*   `Sin` (Orthographic)

### Initialization

You can initialize a projection instance directly by providing the reference pixel (`crpix`), the transformation matrix (`cd`), and the reference sky value (`crval`).

```python
import numpy as np
from mapprojax import Tan, Sin

# Define WCS parameters
crpix = [100.0, 100.0]        # Reference pixel (x, y)
cd_matrix = [[-0.001, 0.0],   # CD matrix (linear transform)
             [0.0, 0.001]]
crval = [45.0, 30.0]          # Reference point (RA, Dec) in degrees

# Create instance for TAN projection
wcs = Tan(crpix, cd_matrix, crval)

# Create instance for SIN projection
# wcs_sin = Sin(crpix, cd_matrix, crval)
```

### Proj (Sky -> Pixel)

The `proj` method converts Right Ascension and Declination (in radians) to Pixel Coordinates.

**Method Signature:**
`proj(ra, dec) -> (x, y)`

**Arguments:**
*   `ra`: Right Ascension in radians (scalar, list, or numpy array).
*   `dec`: Declination in radians (scalar, list, or numpy array).

**Returns:**
*   `x`: Pixel X coordinate.
*   `y`: Pixel Y coordinate.

#### Examples

**1. Single Point**
```python
ra, dec = np.radians(45.1), np.radians(30.1)
x, y = wcs.proj(ra, dec)
print(f"Pixel: {x}, {y}")
```

**2. Array of Points (1D)**
```python
ra_list = np.radians([45.0, 45.1, 45.2])
dec_list = np.radians([30.0, 30.1, 30.2])

x, y = wcs.proj(ra_list, dec_list)
# x and y are numpy arrays of shape (3,)
```

**3. 2D Array of Points**
```python
# Grid of coordinates
ra_grid = np.radians([[45.0, 45.1], [45.0, 45.1]])
dec_grid = np.radians([[30.0, 30.0], [30.1, 30.1]])

x, y = wcs.proj(ra_grid, dec_grid)
# x and y are numpy arrays of shape (2, 2)
```

### Unproj (Pixel -> Sky)

The `unproj` method converts Pixel Coordinates to Right Ascension and Declination (in radians).

**Method Signature:**
`unproj(x, y) -> (ra, dec)`

**Arguments:**
*   `x`: Pixel X coordinate.
*   `y`: Pixel Y coordinate.

**Returns:**
*   `ra`: Right Ascension in radians.
*   `dec`: Declination in radians.

**Example:**
```python
x = [100, 110, 120]
y = [100, 110, 120]
ra_rad, dec_rad = wcs.unproj(x, y)
ra_deg = np.degrees(ra_rad)
```

---

## 2. WCS Arrays (Batch)

For processing multiple WCS configurations simultaneously (sharing `crpix` and `cd` but with different `crval`), use the array variants: `TanArray` and `SinArray`.

### Initialization

These classes take arrays for `crval` (Right Ascension and Declination) instead of single values.

```python
from mapprojax import TanArray, SinArray

# Shared parameters
crpix = [100.0, 100.0]
cd = [[-0.001, 0.0], [0.0, 0.001]]

# Unique parameters for each WCS instance
# Example: 3 WCS instances centered at different RA/Dec
crval_ra = [45.0, 50.0, 55.0]
crval_dec = [30.0, 30.0, 30.0]
crvals = (crval_ra, crval_dec) # Tuple of arrays or (N, 2) array

# Create TanArray
wcs_array = TanArray(crpix, cd, crvals)
```

### Broadcasting Behavior

The `proj` and `unproj` methods support numpy-style broadcasting, combining the shape of the WCS array with the shape of the input coordinates.

**Scenario A: 1-to-1 Mapping**
If you have N WCS instances and provide N coordinates, the operation is element-wise.

```python
# 3 WCS instances, 3 coordinate points
ra = np.radians([45.1, 50.1, 55.1])
dec = np.radians([30.1, 30.1, 30.1])

x, y = wcs_array.proj(ra, dec)
# Result: x and y have shape (3,)
```

**Scenario B: One Coordinate, Multiple WCSs**
Project a single sky point into all WCS frames.

```python
# 1 coordinate point
ra = np.radians(45.0)
dec = np.radians(30.0)

x, y = wcs_array.proj(ra, dec)
# Result: x and y have shape (3,)
```

**Scenario C: All Coordinates to All WCSs (Outer Product)**
To project M points into N WCSs, use reshaping to trigger broadcasting.

```python
# 3 WCS instances -> shape (3, 1)
# 5 points -> shape (1, 5)

ra_points = np.radians(np.array([45, 46, 47, 48, 49]))
dec_points = np.radians(np.full(5, 30.0))

# Reshape input for broadcasting
# wcs_array internal crvals are shape (3,)
# We pass inputs as shape (1, 5)
x, y = wcs_array.proj(ra_points[None, :], dec_points[None, :])
# x, y shape is (3, 5)
```

---

## 3. Serialization

The library supports efficient binary serialization using the ASDF (Advanced Scientific Data Format) standard. This is the recommended method for saving WCS objects to disk.

### ASDF Serialization

Requires the `asdf` library.

```python
import asdf

# Saving to ASDF
tree = {'my_tan': wcs, 'my_tan_array': wcs_array}
with asdf.AsdfFile(tree) as af:
    af.write_to('my_wcs_data.asdf')

# Loading from ASDF
with asdf.open('my_wcs_data.asdf') as af:
    loaded_wcs = af['my_tan']
    loaded_array = af['my_tan_array']
```

### Custom Binary Format (Advanced)

For specialized use cases requiring raw binary access without external dependencies, the library provides a low-level interface.

*   `to_bytes()`: Returns a bytes object containing the serialized data.
*   `from_bytes(data)`: Class method to reconstruct the object from bytes.

**Note:** This is a raw binary dump of the internal state (including NumPy byte buffers) and is intended for efficient inter-process communication or custom storage solutions.