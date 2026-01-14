import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from mapprojax.jax_projections import SinJax, TanJax
from mapprojax.adaptive import adaptive_reproject
import time

def main():
    # 1. Load data and headers
    input_fits = '1904-66_SIN.fits'
    template_fits = '1904-66_TAN_template.fits'
    
    print(f"Loading {input_fits}...")
    with fits.open(input_fits) as hdul:
        data_in = jnp.array(hdul[0].data)
        header_in = hdul[0].header
        wcs_in_astropy = WCS(header_in)

    print(f"Loading template {template_fits}...")
    with fits.open(template_fits) as hdul:
        header_out = hdul[0].header
        wcs_out_astropy = WCS(header_out)

    # 2. Setup mapprojax WCS parameters
    # mapprojax expects crpix (0-based), cd matrix, crval (degrees)
    
    def get_mapprojax_params(w):
        # astropy wcs.wcs.crpix is 1-based
        crpix = jnp.array(w.wcs.crpix - 1.0)
        # pixel_scale_matrix handles PC and CDELT
        cd = jnp.array(w.pixel_scale_matrix)
        crval = jnp.array(w.wcs.crval)
        return crpix, cd, crval

    crpix_in, cd_in, crval_in = get_mapprojax_params(wcs_in_astropy)
    crpix_out, cd_out, crval_out = get_mapprojax_params(wcs_out_astropy)

    # Instantiate JAX-compatible WCS
    wcs_in = SinJax(crpix_in, cd_in, crval_in)
    wcs_out = TanJax(crpix_out, cd_out, crval_out)
    
    # Optimization: Pre-compute combined rotation matrix
    # M_total = M_in @ M_out^T
    # This maps directly from Native_out -> Native_in
    m_combined = jnp.matmul(wcs_in.r_matrix, wcs_out.r_matrix.T)
    
    from mapprojax.utils import apply_rotation

    # 3. Define the transform function (target pixel -> source pixel)
    # This function will be traced by JAX
    def transform(x_out, y_out):
        # 1. Target Pixel -> Target Native
        v_out = wcs_out.pix_to_native(x_out, y_out)
        
        # 2. Rotate Target Native -> Source Native
        # v_in = M_combined * v_out
        v_in = apply_rotation(m_combined, *v_out)
        
        # 3. Source Native -> Source Pixel
        return wcs_in.native_to_pix(*v_in)

    # 4. Run adaptive reprojection
    output_shape = (header_out['NAXIS2'], header_out['NAXIS1'])
    print(f"Reprojecting to shape {output_shape}...")
    
    start_time = time.time()
    reprojected_data = adaptive_reproject(
        data_in,
        transform,
        output_shape,
        kernel='gaussian',
        kernel_width=1.0,
        sample_region_width=4.0,
        max_window_radius=20
    )
    # Trigger execution (JAX is lazy)
    reprojected_data.block_until_ready()
    end_time = time.time()
    
    print(f"Reprojection complete in {end_time - start_time:.2f} seconds.")

    # 5. Save the result
    output_filename = 'reprojected_adaptive.fits'
    # Copy header and update WCS
    new_header = header_out.copy()
    new_header['HISTORY'] = 'Reprojected using mapprojax adaptive'
    
    # Convert back to numpy for fits writing
    fits.writeto(output_filename, np.array(reprojected_data), new_header, overwrite=True)
    print(f"Saved result to {output_filename}")

if __name__ == "__main__":
    main()
