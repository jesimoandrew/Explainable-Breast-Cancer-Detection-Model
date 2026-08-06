"""Convert CBIS-DDSM DICOM images to PNG while preserving folder structure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import pydicom
from pydicom.dataset import FileDataset
from pydicom.pixel_data_handlers.util import apply_voi_lut


BASE_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = BASE_DIR / "cbis_ddsm"
OUTPUT_ROOT = BASE_DIR / "cbis_ddsm_png"


def _as_os_path(path: Path) -> str:
	resolved = path.resolve()
	if os.name == "nt":
		path_text = str(resolved)
		if path_text.startswith("\\\\?\\"):
			return path_text
		return rf"\\?\{path_text}"
	return str(resolved)


def _plain_path(path: Path) -> Path:
	path_text = str(path)
	if path_text.startswith("\\\\?\\"):
		return Path(path_text[4:])
	return path


def _to_uint8(array: np.ndarray) -> np.ndarray:
	array = np.asarray(array)

	if array.dtype == np.uint8:
		return array

	array = array.astype(np.float32)
	min_value = float(np.min(array))
	max_value = float(np.max(array))

	if max_value <= min_value:
		return np.zeros(array.shape, dtype=np.uint8)

	scaled = (array - min_value) / (max_value - min_value)
	return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


def _extract_pixels(dataset: FileDataset) -> np.ndarray:
	pixel_array = dataset.pixel_array

	if getattr(dataset, "PhotometricInterpretation", None) == "MONOCHROME1":
		pixel_array = np.max(pixel_array) - pixel_array

	try:
		pixel_array = apply_voi_lut(pixel_array, dataset)
	except Exception:
		pass

	return _to_uint8(pixel_array)


def convert_dicom_to_png(dicom_path: Path, source_root: Path, output_root: Path) -> Path:
	dicom_path = _plain_path(dicom_path).resolve()
	source_root = _plain_path(source_root).resolve()
	output_root = _plain_path(output_root).resolve()

	relative_path = dicom_path.relative_to(source_root)
	output_path = output_root / relative_path.with_suffix(".png")
	
	# Skip if PNG already exists
	if output_path.exists():
		return output_path
	
	os.makedirs(_as_os_path(output_path.parent), exist_ok=True)

	dataset = pydicom.dcmread(_as_os_path(dicom_path))
	image_array = _extract_pixels(dataset)
	image = Image.fromarray(image_array)
	image.save(_as_os_path(output_path), format="PNG", compress_level=1)
	return output_path


def iter_dicom_files(source_root: Path) -> Iterable[Path]:
	for root, _, files in os.walk(_as_os_path(source_root)):
		for file_name in files:
			if file_name.lower().endswith(".dcm"):
				yield Path(root) / file_name


def precreate_output_directories(source_root: Path, output_root: Path) -> int:
	created = 0
	source_root = _plain_path(source_root).resolve()
	output_root = _plain_path(output_root).resolve()

	for root, dirs, _ in os.walk(_as_os_path(source_root)):
		root_path = _plain_path(Path(root)).resolve()
		relative_root = root_path.relative_to(source_root)
		target_root = output_root / relative_root
		os.makedirs(_as_os_path(target_root), exist_ok=True)
		created += 1

		for directory in dirs:
			target_dir = target_root / directory
			os.makedirs(_as_os_path(target_dir), exist_ok=True)

	return created


def remove_output_tree(output_root: Path) -> None:
	output_root = _plain_path(output_root).resolve()
	if not output_root.exists():
		return

	for root, dirs, files in os.walk(_as_os_path(output_root), topdown=False):
		for file_name in files:
			os.remove(Path(root) / file_name)
		for directory in dirs:
			os.rmdir(Path(root) / directory)

	os.rmdir(output_root)


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Convert CBIS-DDSM DICOM images into PNGs while preserving folder structure."
	)
	parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	args = parser.parse_args()

	source_root = args.source_root.resolve()
	output_root = args.output_root.resolve()

	if not source_root.exists():
		raise FileNotFoundError(f"Source root does not exist: {source_root}")

	directory_count = precreate_output_directories(source_root, output_root)
	print(f"prepared directories: {directory_count} source folders mirrored to {output_root}")

	dicom_files = list(iter_dicom_files(source_root))
	total = len(dicom_files)

	converted = 0
	failed = 0

	for index, dicom_path in enumerate(dicom_files, start=1):
		try:
			convert_dicom_to_png(dicom_path, source_root, output_root)
			converted += 1
			if index <= 10 or index % 100 == 0 or index == total:
				print(f"progress: {index}/{total} converted: {dicom_path}")
		except Exception as exc:
			failed += 1
			print(f"failed: {dicom_path} -> {exc}")

	print(
		f"done: converted={converted}, failed={failed}, output_root={output_root.resolve()}"
	)
	return 0 if failed == 0 else 1


if __name__ == "__main__":
	raise SystemExit(main())
