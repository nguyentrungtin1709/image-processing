"""
Batch Label Detection Script

This script processes all images in the input folder, detects labels using
the LabelDetector class, and saves the results to the output folder along
with CSV reports.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for importing core module
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.label_detector import DEFAULT_CONFIG, LabelDetector


# Supported image extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def create_detector(config: dict) -> LabelDetector:
    """
    Create a LabelDetector instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary with all required parameters.
        
    Returns:
        Configured LabelDetector instance.
    """
    return LabelDetector(
        thresh_min=config["thresh_min"],
        thresh_max=config["thresh_max"],
        thresh_step=config["thresh_step"],
        min_area_ratio=config["min_area_ratio"],
        max_area_ratio=config["max_area_ratio"],
        approx_epsilon=config["approx_epsilon"],
        min_aspect_ratio=config["min_aspect_ratio"],
        max_aspect_ratio=config["max_aspect_ratio"],
        group_threshold=config["group_threshold"],
        group_eps=config["group_eps"],
        box_color=config["box_color"],
        box_thickness=config["box_thickness"],
        font_scale=config["font_scale"],
        font_thickness=config["font_thickness"],
        time_font_scale=config["time_font_scale"],
        time_font_thickness=config["time_font_thickness"],
        time_bg_color=config["time_bg_color"],
        time_text_color=config["time_text_color"],
    )


def get_image_files(input_folder: Path) -> list:
    """
    Get all supported image files from the input folder.
    
    Args:
        input_folder: Path to input folder.
        
    Returns:
        List of image file paths.
    """
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_folder.glob(f"*{ext}"))
        image_files.extend(input_folder.glob(f"*{ext.upper()}"))
    return sorted(set(image_files))


def process_image(
    detector: LabelDetector,
    image_path: Path,
    output_folder: Path
) -> dict:
    """
    Process a single image and save the result.
    
    Args:
        detector: LabelDetector instance.
        image_path: Path to input image.
        output_folder: Path to output folder.
        
    Returns:
        Dictionary with processing results.
    """
    # Read image
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"Warning: Could not read image: {image_path.name}")
        return None
    
    # Detect labels
    boxes, elapsed_ms = detector.detect(image)
    
    # Draw results
    result_image = detector.draw_results(image, boxes, elapsed_ms)
    
    # Save output image
    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), result_image)
    
    return {
        "image_name": image_path.name,
        "num_labels": len(boxes),
        "time_ms": elapsed_ms
    }


def write_details_csv(results: list, output_path: Path) -> None:
    """
    Write detailed results for each image to CSV.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to output CSV file.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["image_name", "num_labels", "time_ms"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                "image_name": result["image_name"],
                "num_labels": result["num_labels"],
                "time_ms": f"{result['time_ms']:.2f}"
            })


def write_summary_csv(results: list, output_path: Path) -> None:
    """
    Write summary statistics to CSV.
    
    Args:
        results: List of result dictionaries.
        output_path: Path to output CSV file.
    """
    if not results:
        return
    
    # Calculate statistics
    total_images = len(results)
    total_labels = sum(r["num_labels"] for r in results)
    times = [r["time_ms"] for r in results]
    
    avg_labels = total_labels / total_images if total_images > 0 else 0
    avg_time = np.mean(times) if times else 0
    std_time = np.std(times) if times else 0
    min_time = np.min(times) if times else 0
    max_time = np.max(times) if times else 0
    
    # Write summary
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_images", total_images])
        writer.writerow(["total_labels", total_labels])
        writer.writerow(["avg_labels_per_image", f"{avg_labels:.2f}"])
        writer.writerow(["avg_time_ms", f"{avg_time:.2f}"])
        writer.writerow(["std_time_ms", f"{std_time:.2f}"])
        writer.writerow(["min_time_ms", f"{min_time:.2f}"])
        writer.writerow(["max_time_ms", f"{max_time:.2f}"])


def main():
    """Main entry point for batch label detection."""
    # Get project root directory
    project_root = Path(__file__).parent.parent
    
    # Define input and output folders
    input_folder = project_root / "input"
    output_folder = project_root / "output"
    report_folder = output_folder / "report"
    
    # Create output folders if not exists
    output_folder.mkdir(parents=True, exist_ok=True)
    report_folder.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Create detector
    detector = create_detector(config)
    
    # Get all image files
    image_files = get_image_files(input_folder)
    
    if not image_files:
        print(f"No images found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("-" * 50)
    
    # Process each image
    results = []
    for image_path in image_files:
        result = process_image(detector, image_path, output_folder)
        if result:
            results.append(result)
            print(f"Processed: {result['image_name']} | "
                  f"Labels: {result['num_labels']} | "
                  f"Time: {result['time_ms']:.2f} ms")
    
    print("-" * 50)
    
    # Write CSV reports
    if results:
        details_path = report_folder / "details.csv"
        summary_path = report_folder / "summary.csv"
        
        write_details_csv(results, details_path)
        write_summary_csv(results, summary_path)
        
        print(f"Details saved to: {details_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"Total images processed: {len(results)}")
    else:
        print("No images were processed successfully")


if __name__ == "__main__":
    main()
