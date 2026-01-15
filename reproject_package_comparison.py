import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_adaptive

def main():
    input_file = "reproj_test/level2_2025W23_1C_0165_1D4_spx_l2b-v20-2025-251.fits"
    template_file = "reproj_test/template.fits"
    output_file = "reprojected_package.fits"

    print(f"Loading input: {input_file}")
    with fits.open(input_file) as hdul_in:
        # Extract the data and WCS while the file is open
        data_in = hdul_in[1].data
        wcs_in = WCS(hdul_in[1].header)

    print(f"Loading template: {template_file}")
    with fits.open(template_file) as hdul_tmpl:
        # The template can be specified by its header
        header_out = hdul_tmpl[0].header
        shape_out = (header_out['NAXIS2'], header_out['NAXIS1'])

    print(f"Reprojecting to shape {shape_out} using the 'reproject' package...")
    
    # reproject_adaptive automatically handles coordinate system conversions
    # (e.g., ICRS to Galactic) based on the FITS headers.
    # It also handles SIP distortions present in the input WCS.
    import time
    t0 = time.time()
    for i in range(10):
        print(f"Iteration {i}...")
        array_out, footprint = reproject_adaptive(
            (data_in, wcs_in),
            header_out,
            shape_out=shape_out
        )
    t1 = time.time()
    
    total_time = t1 - t0
    print(f"\nReprojection complete.")
    print(f"Total time for 10 iterations: {total_time:.3f}s")
    print(f"Average time per iteration: {total_time/10:.3f}s")

    # Save the output
    hdu_out = fits.PrimaryHDU(data=array_out, header=header_out)
    hdu_out.writeto(output_file, overwrite=True)
    print(f"Saved reprojected image to {output_file}")
    
    # Print some stats
    valid_pixels = np.isfinite(array_out)
    print(f"Output NaNs: {np.sum(~valid_pixels)} / {array_out.size} ({np.sum(~valid_pixels)/array_out.size:.1%})")
    if np.sum(valid_pixels) > 0:
        print(f"Output Min/Max: {np.nanmin(array_out)} / {np.nanmax(array_out)}")


if __name__ == "__main__":
    main()
