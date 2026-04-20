from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


WORKSPACE_DIR = Path(__file__).resolve().parent
HERCULENS_DIR = WORKSPACE_DIR.parent.parent
DATA_DIR = HERCULENS_DIR / "Data" / "BensLens"
OUTPUT_DIR = WORKSPACE_DIR / "preprocessed" / "prep_20260420_benslens_v1"

CUTOUT_SIZE = 220
REFERENCE_CENTER_F814W = (2055, 1151)
FILE_MAP = {
    "f475w": DATA_DIR / "ifo611010_drc.fits",
    "f814w": DATA_DIR / "ifo612010_drc.fits",
}


def pixel_scale_arcsec(header: fits.Header) -> float:
    if "PIXSCALE" in header:
        return float(header["PIXSCALE"])
    if "D001SCAL" in header:
        return float(header["D001SCAL"])
    cd11 = header.get("CD1_1")
    cd21 = header.get("CD2_1")
    if cd11 is not None and cd21 is not None:
        return float(np.hypot(cd11, cd21) * 3600.0)
    raise ValueError("Cannot determine pixel scale from FITS header.")


def load_sci_image(path: Path) -> tuple[np.ndarray, fits.Header, WCS]:
    with fits.open(path, memmap=True) as hdul:
        sci_index = 1 if len(hdul) > 1 and hdul[1].data is not None else 0
        data = np.asarray(hdul[sci_index].data, dtype=np.float64)
        primary_header = hdul[0].header
        header = hdul[sci_index].header.copy()
    for key in ("FILTER", "EXPTIME", "ORIENTAT", "INSTRUME", "DETECTOR"):
        if key not in header and key in primary_header:
            header[key] = primary_header[key]
    return data, header, WCS(header)


def make_integer_cutout(
    data: np.ndarray,
    header: fits.Header,
    wcs: WCS,
    center_xy: tuple[int, int],
    size: int,
) -> tuple[np.ndarray, fits.Header]:
    cutout = Cutout2D(
        data,
        position=(int(center_xy[0]), int(center_xy[1])),
        size=(size, size),
        wcs=wcs,
        mode="strict",
        copy=True,
    )
    cutout_header = header.copy()
    cutout_header.update(cutout.wcs.to_header())
    cutout_header["CUTSIZE"] = size
    cutout_header["CUTXCTR"] = int(center_xy[0])
    cutout_header["CUTYCTR"] = int(center_xy[1])
    cutout_header["CUTNOTE"] = "Integer-center Cutout2D, no interpolation"
    return np.asarray(cutout.data, dtype=np.float64), cutout_header


def write_cutout(path: Path, data: np.ndarray, header: fits.Header) -> None:
    fits.PrimaryHDU(data=data, header=header).writeto(path, overwrite=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data_f814, header_f814, wcs_f814 = load_sci_image(FILE_MAP["f814w"])
    data_f475, header_f475, wcs_f475 = load_sci_image(FILE_MAP["f475w"])

    x814, y814 = map(float, REFERENCE_CENTER_F814W)
    ra_deg, dec_deg = wcs_f814.pixel_to_world_values(x814, y814)
    x475_float, y475_float = wcs_f475.world_to_pixel_values(ra_deg, dec_deg)

    center_f814_int = (int(np.rint(x814)), int(np.rint(y814)))
    center_f475_int = (int(np.rint(x475_float)), int(np.rint(y475_float)))

    cutout_f814, cutout_header_f814 = make_integer_cutout(
        data_f814,
        header_f814,
        wcs_f814,
        center_f814_int,
        CUTOUT_SIZE,
    )
    cutout_f475, cutout_header_f475 = make_integer_cutout(
        data_f475,
        header_f475,
        wcs_f475,
        center_f475_int,
        CUTOUT_SIZE,
    )

    output_f814 = OUTPUT_DIR / "f814w_lens_cutout_220x220.fits"
    output_f475 = OUTPUT_DIR / "f475w_lens_cutout_220x220.fits"
    write_cutout(output_f814, cutout_f814, cutout_header_f814)
    write_cutout(output_f475, cutout_f475, cutout_header_f475)

    metadata = {
        "cutout_size": CUTOUT_SIZE,
        "science_cutout_size": [CUTOUT_SIZE, CUTOUT_SIZE],
        "reference_band": "f814w",
        "reference_center_f814w_pix_input": [x814, y814],
        "reference_center_f814w_pix_integer": list(center_f814_int),
        "mapped_center_f475w_pix_float": [float(x475_float), float(y475_float)],
        "mapped_center_f475w_pix_integer": list(center_f475_int),
        "reference_center_world_deg": {
            "ra": float(ra_deg),
            "dec": float(dec_deg),
        },
        "bands": {
            "f814w": {
                "input_path": str(FILE_MAP["f814w"]),
                "output_cutout_path": str(output_f814),
                "center_pix_integer": list(center_f814_int),
                "filter": header_f814.get("FILTER", "F814W"),
                "pixel_scale_arcsec": pixel_scale_arcsec(header_f814),
            },
            "f475w": {
                "input_path": str(FILE_MAP["f475w"]),
                "output_cutout_path": str(output_f475),
                "center_pix_float_from_wcs": [float(x475_float), float(y475_float)],
                "center_pix_integer": list(center_f475_int),
                "filter": header_f475.get("FILTER", "F475W"),
                "pixel_scale_arcsec": pixel_scale_arcsec(header_f475),
            },
        },
        "notes": [
            "F475W center is mapped from the F814W reference center through WCS.",
            "Cutout centers are rounded to integer pixels before extraction.",
            "No interpolation or resampling is applied.",
        ],
    }
    metadata_json = json.dumps(metadata, indent=2)
    (OUTPUT_DIR / "benslens_cutout_metadata.json").write_text(metadata_json)
    (OUTPUT_DIR / "preprocessing_metadata.json").write_text(metadata_json)

    print("BensLens cutouts written:")
    print("  F814W:", output_f814)
    print("  F475W:", output_f475)
    print("  Metadata:", OUTPUT_DIR / "benslens_cutout_metadata.json")
    print("  Preprocess metadata:", OUTPUT_DIR / "preprocessing_metadata.json")
    print("  F814W center integer:", center_f814_int)
    print("  F475W center float:", (float(x475_float), float(y475_float)))
    print("  F475W center integer:", center_f475_int)


if __name__ == "__main__":
    main()
