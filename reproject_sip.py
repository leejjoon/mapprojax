import numpy as np
import jax
# Enable 64-bit precision for accurate WCS (must be before mapprojax imports)
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.wcs import WCS
from mapprojax.wcs_utils import load_wcs_params_from_header
from mapprojax.jax_projections import TanJax
from mapprojax.adaptive import adaptive_reproject
from mapprojax.utils import apply_rotation

def get_frame_rotation_matrix(header_from, header_to):
    """
    Compute 3x3 rotation matrix to convert Cartesian vectors from the 'from' frame to the 'to' frame.
    Uses Astropy to handle the frames.
    """
    # Create dummy WCS objects just to get the frame (ignoring SIP/distortions for frame determination)
    wcs_from = WCS(header_from)
    wcs_to = WCS(header_to)
    
    frame_from = wcs_to_celestial_frame(wcs_from)
    frame_to = wcs_to_celestial_frame(wcs_to)
    
    print(f"Converting coordinates from {frame_from.name} to {frame_to.name}")
    
    if frame_from.name == frame_to.name:
        return np.eye(3)
        
    # Compute rotation matrix by transforming basis vectors
    basis_lon = [0, 90, 0]
    basis_lat = [0, 0, 90]
    
    coords = SkyCoord(basis_lon, basis_lat, unit='deg', frame=frame_from)
    coords_transformed = coords.transform_to(frame_to)
    cart = coords_transformed.cartesian
    
    # Transformed basis vectors must be columns for matrix-vector multiplication
    # np.stack(..., axis=0) results in rows being [X_components, Y_components, Z_components]
    # which means columns are the transformed basis vectors [x, y, z]^T.
    matrix = np.stack([cart.x.value, cart.y.value, cart.z.value], axis=0)
    
    return matrix

def main():
    input_file = "reproj_test/level2_2025W23_1C_0165_1D4_spx_l2b-v20-2025-251.fits"
    template_file = "reproj_test/template.fits"
    output_file = "reprojected_sip.fits"

    print(f"Loading input: {input_file}")
    with fits.open(input_file) as hdul_in:
        # Extension 1 has image data
        if len(hdul_in) > 1 and hdul_in[1].data is not None:
            data_in = hdul_in[1].data
            header_in = hdul_in[1].header
        else:
            data_in = hdul_in[0].data
            header_in = hdul_in[0].header
            
    print(f"Loading template: {template_file}")
    with fits.open(template_file) as hdul_tmpl:
        if len(hdul_tmpl) > 1 and hdul_tmpl[1].data is not None:
             header_out = hdul_tmpl[1].header
             shape_out = hdul_tmpl[1].data.shape
        else:
             header_out = hdul_tmpl[0].header
             if hdul_tmpl[0].data is not None:
                 shape_out = hdul_tmpl[0].data.shape
             else:
                 shape_out = (header_out['NAXIS2'], header_out['NAXIS1'])

    # 1. Setup Input WCS
    crpix_in, cd_in, crval_in, sip_in = load_wcs_params_from_header(header_in)
    wcs_in = TanJax(crpix_in, cd_in, crval_in, sip=sip_in)
    print("Input WCS created (with SIP)." if sip_in else "Input WCS created (no SIP).")

    # 2. Setup Output WCS
    crpix_out, cd_out, crval_out, sip_out = load_wcs_params_from_header(header_out)
    wcs_out = TanJax(crpix_out, cd_out, crval_out, sip=sip_out)
    print("Output WCS created (with SIP)." if sip_out else "Output WCS created (no SIP).")

    # 3. Setup Coordinate Conversion
    rot_matrix_np = get_frame_rotation_matrix(header_out, header_in)
    rot_matrix = jnp.array(rot_matrix_np) # Move to JAX

    # 4. Define Transformation Function (Output Pix -> Input Pix)
    def transform(x_out, y_out):
        # Output Pix -> Output Unit Vector (Galactic)
        v_out = wcs_out.unproj_xyz(x_out, y_out)
        # Rotation (Galactic -> ICRS)
        v_in = apply_rotation(rot_matrix, *v_out)
        # Input Unit Vector (ICRS) -> Input Pix
        return wcs_in.proj_xyz(*v_in)

    # 5. Prepare Data
    image_in = jnp.array(data_in, dtype=float)

    print(f"Reprojecting to shape {shape_out}...")
    
    # 6. Run Reprojection
    import time
    t0 = time.time()
    for i in range(10):
        print(f"Iteration {i}...")
        image_out = adaptive_reproject(
            image_in,
            transform,
            output_shape=shape_out,
            kernel='gaussian',
            block_size=None,
            max_window_radius=5
        )
        # For JAX timing, we must block until the async execution is done
        image_out.block_until_ready()
        
        if i == 0:
            t1 = time.time()
            print(f"  First iteration (includes JIT): {t1 - t0:.3f}s")
            t_start_loop = time.time()

    t_end = time.time()
    
    total_time_9 = t_end - t_start_loop
    print(f"\nReprojection complete.")
    print(f"Average time for subsequent 9 iterations: {total_time_9/9:.3f}s")
    print(f"Total time for 10 iterations: {t_end - t0:.3f}s")

    image_out_np = np.array(image_out)
    # 7. Save Output
    hdu_out = fits.PrimaryHDU(data=image_out_np, header=header_out)
    hdu_out.writeto(output_file, overwrite=True)
    print(f"Saved reprojected image to {output_file}")

if __name__ == "__main__":
    main()
