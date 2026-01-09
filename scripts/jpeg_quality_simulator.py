"""
JPEG Quality Simulator
======================
Simulate low-quality images using JPEG compression technique.

Logic:
    1. Read original image
    2. Encode image to JPEG with specified quality level (10%, 50%)
    3. Decode back to image
    4. Save compressed image to corresponding output folder

Usage:
    python scripts/jpeg_quality_simulator.py --input <input_folder> --output <output_folder>
    python scripts/jpeg_quality_simulator.py  # Default: input -> output
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# List of JPEG quality levels to generate (0-100)
# 100 = highest quality, 10 = low quality (heavy compression)
JPEG_QUALITY_LEVELS: List[int] = [50, 10]

# Supported image file extensions
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CORE PROCESSING
# ============================================================================

class JpegQualitySimulator:
    """
    Class for simulating low JPEG quality.
    
    Follows Single Responsibility Principle (SRP):
    - Only responsible for JPEG compression/decompression
    """
    
    def __init__(self, quality_levels: Optional[List[int]] = None):
        """
        Initialize simulator with quality levels.
        
        Args:
            quality_levels: List of JPEG quality levels (0-100).
                           Default: [50, 10]
        """
        self._quality_levels = quality_levels or JPEG_QUALITY_LEVELS
        self._validate_quality_levels()
    
    def _validate_quality_levels(self) -> None:
        """Validate quality levels are within valid range."""
        for level in self._quality_levels:
            if not 0 <= level <= 100:
                raise ValueError(
                    f"Quality level must be between 0-100, got: {level}"
                )
    
    @property
    def quality_levels(self) -> List[int]:
        """Return list of quality levels."""
        return self._quality_levels.copy()
    
    def compress_image(self, image: np.ndarray, quality: int) -> np.ndarray:
        """
        Compress image using JPEG with specified quality level.
        
        Logic equivalent to C# OpenCvSharp:
            Mat compressed = new Mat();
            var encodeParams = new[] { new ImageEncodingParam(ImwriteFlags.JpegQuality, quality) };
            Cv2.ImEncode(".jpg", mat, out byte[] jpegData, encodeParams);
            compressed = Cv2.ImDecode(jpegData, ImreadModes.Color);
        
        Args:
            image: Input image (numpy array BGR)
            quality: JPEG quality level (0-100)
        
        Returns:
            Compressed image (numpy array BGR)
        """
        # Step 1: Encode image to JPEG with quality level
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        success, jpeg_data = cv2.imencode(".jpg", image, encode_params)
        
        if not success:
            raise RuntimeError(f"Failed to encode image with quality={quality}")
        
        # Step 2: Decode JPEG data back to image
        compressed = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
        
        if compressed is None:
            raise RuntimeError("Failed to decode JPEG data")
        
        return compressed


class ImageFileHandler:
    """
    Class for handling image file I/O operations.
    
    Follows Single Responsibility Principle (SRP):
    - Only responsible for image file I/O
    """
    
    def __init__(self, supported_extensions: Optional[Tuple[str, ...]] = None):
        """
        Initialize handler with supported extensions.
        
        Args:
            supported_extensions: Tuple of supported file extensions
        """
        self._supported_extensions = supported_extensions or SUPPORTED_EXTENSIONS
    
    def is_supported_image(self, file_path: Path) -> bool:
        """Check if file is a supported image format."""
        return file_path.suffix.lower() in self._supported_extensions
    
    def scan_directory(self, directory: Path) -> List[Path]:
        """
        Scan directory and return list of image files.
        
        Args:
            directory: Path to directory to scan
        
        Returns:
            List of supported image files
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")
        
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        image_files = [
            f for f in directory.iterdir()
            if f.is_file() and self.is_supported_image(f)
        ]
        
        return sorted(image_files)
    
    def read_image(self, file_path: Path) -> np.ndarray:
        """
        Read image from file.
        
        Args:
            file_path: Path to image file
        
        Returns:
            Image as numpy array (BGR)
        """
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        
        if image is None:
            raise IOError(f"Failed to read image: {file_path}")
        
        return image
    
    def save_image(self, image: np.ndarray, file_path: Path) -> None:
        """
        Save image to file.
        
        Args:
            image: Image to save (numpy array BGR)
            file_path: Output file path
        """
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        success = cv2.imwrite(str(file_path), image)
        
        if not success:
            raise IOError(f"Failed to save image: {file_path}")


class BatchProcessor:
    """
    Class for coordinating batch image processing.
    
    Follows:
    - Single Responsibility Principle (SRP): Workflow coordination
    - Dependency Inversion Principle (DIP): Depends on abstractions
    """
    
    def __init__(
        self,
        simulator: JpegQualitySimulator,
        file_handler: ImageFileHandler
    ):
        """
        Initialize processor with dependencies.
        
        Args:
            simulator: JPEG compression processor object
            file_handler: File I/O handler object
        """
        self._simulator = simulator
        self._file_handler = file_handler
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> dict:
        """
        Process all images in directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Root output directory
        
        Returns:
            Dictionary containing processing statistics
        """
        stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "quality_levels": self._simulator.quality_levels
        }
        
        # Scan input directory
        image_files = self._file_handler.scan_directory(input_dir)
        stats["total"] = len(image_files)
        
        if stats["total"] == 0:
            logger.warning(f"No images found in: {input_dir}")
            return stats
        
        logger.info(f"Found {stats['total']} images in {input_dir}")
        logger.info(f"Quality levels: {self._simulator.quality_levels}")
        
        # Process each image
        for image_file in image_files:
            try:
                self._process_single_image(image_file, output_dir)
                stats["success"] += 1
                logger.info(f"[OK] Processed: {image_file.name}")
            except Exception as e:
                stats["failed"] += 1
                logger.error(f"[FAILED] Error processing {image_file.name}: {e}")
        
        return stats
    
    def _process_single_image(self, image_path: Path, output_dir: Path) -> None:
        """
        Process a single image with all quality levels.
        
        Args:
            image_path: Path to input image
            output_dir: Root output directory
        """
        # Read image
        image = self._file_handler.read_image(image_path)
        
        # Process with each quality level
        for quality in self._simulator.quality_levels:
            # Compress image
            compressed = self._simulator.compress_image(image, quality)
            
            # Create output path: output/{quality}/{filename}
            output_subdir = output_dir / str(quality)
            output_path = output_subdir / image_path.name
            
            # Save image
            self._file_handler.save_image(compressed, output_path)
            
            logger.debug(f"  Saved quality={quality}: {output_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate low-quality images using JPEG compression technique",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/jpeg_quality_simulator.py
    python scripts/jpeg_quality_simulator.py --input ./my_images --output ./results
    python scripts/jpeg_quality_simulator.py --quality 30 50 70
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="input",
        help="Directory containing input images (default: input)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--quality", "-q",
        type=int,
        nargs="+",
        default=[50, 10],
        help="JPEG quality levels (0-100), default: 50 10"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main function to orchestrate the entire program.
    
    Returns:
        Exit code (0 = success, 1 = error)
    """
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine paths
    # Get project root directory (parent of scripts folder)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # If path is relative, convert to absolute from project root
    if not input_dir.is_absolute():
        input_dir = project_root / input_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    
    logger.info("=" * 60)
    logger.info("JPEG Quality Simulator")
    logger.info("=" * 60)
    logger.info(f"Input directory : {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Quality levels  : {args.quality}")
    logger.info("=" * 60)
    
    try:
        # Initialize components (Dependency Injection)
        simulator = JpegQualitySimulator(quality_levels=args.quality)
        file_handler = ImageFileHandler()
        processor = BatchProcessor(simulator, file_handler)
        
        # Process
        stats = processor.process_directory(input_dir, output_dir)
        
        # Display results
        logger.info("=" * 60)
        logger.info("PROCESSING RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total images : {stats['total']}")
        logger.info(f"Successful   : {stats['success']}")
        logger.info(f"Failed       : {stats['failed']}")
        logger.info("=" * 60)
        
        if stats["failed"] > 0:
            logger.warning("Some images failed to process. Check log for details.")
            return 1
        
        if stats["total"] == 0:
            logger.warning("No images were processed.")
            return 1
        
        logger.info("Completed!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
