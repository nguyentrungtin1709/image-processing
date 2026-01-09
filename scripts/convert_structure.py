"""
Script to convert input folder structure to samples format.

This script transforms the input directory structure by:
1. Reading each subdirectory containing images and a JSON label file
2. Copying images to output/samples/
3. Creating corresponding result_{image_name}.json files with the same label data
"""

import json
import logging
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the structure converter."""
    input_dir: Path
    output_dir: Path
    samples_folder_name: str = "samples"
    image_extensions: Set[str] = field(default_factory=lambda: {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"
    })
    result_prefix: str = "result_"
    overwrite_existing: bool = False


class StructureConverter:
    """Converts input folder structure to samples format."""

    def __init__(self, config: Config):
        self._config = config
        self._samples_dir = config.output_dir / config.samples_folder_name
        self._processed_files: List[str] = []
        self._conflicts: List[str] = []

    def convert(self) -> None:
        """Execute the conversion process."""
        logger.info("Starting structure conversion...")
        logger.info(f"Input directory: {self._config.input_dir}")
        logger.info(f"Output directory: {self._samples_dir}")

        self._create_output_directory()
        self._process_all_subdirectories()
        self._log_summary()

    def _create_output_directory(self) -> None:
        """Create the samples output directory if it doesn't exist."""
        self._samples_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {self._samples_dir}")

    def _process_all_subdirectories(self) -> None:
        """Process all subdirectories in the input folder."""
        input_path = self._config.input_dir

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        subdirectories = [d for d in input_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirectories)} subdirectories to process")

        for subdir in subdirectories:
            self._process_subdirectory(subdir)

    def _process_subdirectory(self, subdir: Path) -> None:
        """Process a single subdirectory."""
        logger.info(f"Processing subdirectory: {subdir.name}")

        # Find the JSON label file
        json_file = self._find_json_file(subdir)
        if json_file is None:
            logger.warning(f"No JSON file found in {subdir.name}, skipping...")
            return

        # Read the JSON content
        label_data = self._read_json_file(json_file)
        if label_data is None:
            return

        # Find and process all image files
        image_files = self._find_image_files(subdir)
        logger.info(f"Found {len(image_files)} images in {subdir.name}")

        for image_file in image_files:
            self._process_image(image_file, label_data)

    def _find_json_file(self, subdir: Path) -> Path | None:
        """Find the JSON label file in a subdirectory."""
        expected_json_name = f"{subdir.name}.json"
        expected_json_path = subdir / expected_json_name

        if expected_json_path.exists():
            return expected_json_path

        # Fallback: find any JSON file
        json_files = list(subdir.glob("*.json"))
        if json_files:
            logger.warning(
                f"Expected {expected_json_name}, found {json_files[0].name} instead"
            )
            return json_files[0]

        return None

    def _read_json_file(self, json_path: Path) -> dict | None:
        """Read and parse a JSON file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {json_path}: {e}")
            return None
        except IOError as e:
            logger.error(f"Failed to read file {json_path}: {e}")
            return None

    def _find_image_files(self, subdir: Path) -> List[Path]:
        """Find all image files in a subdirectory."""
        image_files = []
        for file_path in subdir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self._config.image_extensions:
                image_files.append(file_path)
        return sorted(image_files)

    def _process_image(self, image_path: Path, label_data: dict) -> None:
        """Process a single image file."""
        image_name = image_path.name
        image_stem = image_path.stem  # filename without extension

        # Check for conflicts
        dest_image_path = self._samples_dir / image_name
        if dest_image_path.exists() and not self._config.overwrite_existing:
            self._conflicts.append(image_name)
            logger.warning(f"Conflict: {image_name} already exists, skipping...")
            return

        # Copy image file
        self._copy_image(image_path, dest_image_path)

        # Create result JSON file
        result_json_name = f"{self._config.result_prefix}{image_stem}.json"
        result_json_path = self._samples_dir / result_json_name
        self._write_result_json(result_json_path, label_data)

        self._processed_files.append(image_name)

    def _copy_image(self, source: Path, destination: Path) -> None:
        """Copy an image file to the destination."""
        try:
            shutil.copy2(source, destination)
            logger.debug(f"Copied: {source.name} -> {destination}")
        except IOError as e:
            logger.error(f"Failed to copy {source.name}: {e}")
            raise

    def _write_result_json(self, json_path: Path, data: dict) -> None:
        """Write the result JSON file."""
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Created: {json_path.name}")
        except IOError as e:
            logger.error(f"Failed to write {json_path.name}: {e}")
            raise

    def _log_summary(self) -> None:
        """Log a summary of the conversion process."""
        logger.info("=" * 50)
        logger.info("Conversion completed!")
        logger.info(f"Total images processed: {len(self._processed_files)}")
        if self._conflicts:
            logger.warning(f"Conflicts (skipped): {len(self._conflicts)}")
            for conflict in self._conflicts:
                logger.warning(f"  - {conflict}")
        logger.info("=" * 50)


def main():
    """Main entry point for the script."""
    # Get the project root directory (parent of scripts folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Configure paths
    config = Config(
        input_dir=project_root / "input",
        output_dir=project_root / "output",
        samples_folder_name="samples",
        overwrite_existing=False
    )

    # Run the converter
    converter = StructureConverter(config)
    
    try:
        converter.convert()
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())
