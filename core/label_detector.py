"""
Label Detector Module

This module provides the LabelDetector class for detecting labels on t-shirt images
using multi-channel, multi-threshold image processing techniques.
"""

import time
from typing import List, Tuple

import cv2
import numpy as np


# Default configuration for label detection
DEFAULT_CONFIG = {
    # Threshold parameters
    "thresh_min": 180,
    "thresh_max": 241,
    "thresh_step": 20,
    
    # Area filter parameters (percentage of image area)
    "min_area_ratio": 0.005,
    "max_area_ratio": 0.2,
    
    # Shape parameters
    "approx_epsilon": 0.04,
    "min_aspect_ratio": 0.2,
    "max_aspect_ratio": 2.4,
    
    # Grouping parameters
    "group_threshold": 3,
    "group_eps": 0.1,
    
    # Drawing parameters
    "box_color": (0, 255, 0),
    "box_thickness": 3,
    "font_scale": 0.6,
    "font_thickness": 2,
    "time_font_scale": 0.8,
    "time_font_thickness": 2,
    "time_bg_color": (0, 0, 0),
    "time_text_color": (255, 255, 255),
}


class LabelDetector:
    """
    A class for detecting labels on t-shirt images.
    
    This detector uses multi-channel (B, G, R) and multi-threshold analysis
    to find rectangular label regions that appear consistently across
    different processing parameters.
    """
    
    def __init__(
        self,
        thresh_min: int,
        thresh_max: int,
        thresh_step: int,
        min_area_ratio: float,
        max_area_ratio: float,
        approx_epsilon: float,
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        group_threshold: int,
        group_eps: float,
        box_color: Tuple[int, int, int],
        box_thickness: int,
        font_scale: float,
        font_thickness: int,
        time_font_scale: float,
        time_font_thickness: int,
        time_bg_color: Tuple[int, int, int],
        time_text_color: Tuple[int, int, int],
    ):
        """
        Initialize the LabelDetector with configuration parameters.
        
        Args:
            thresh_min: Minimum threshold value for binary thresholding.
            thresh_max: Maximum threshold value for binary thresholding.
            thresh_step: Step size for threshold iteration.
            min_area_ratio: Minimum contour area as ratio of image area.
            max_area_ratio: Maximum contour area as ratio of image area.
            approx_epsilon: Epsilon factor for polygon approximation.
            min_aspect_ratio: Minimum aspect ratio for valid rectangles.
            max_aspect_ratio: Maximum aspect ratio for valid rectangles.
            group_threshold: Minimum occurrences for rectangle grouping.
            group_eps: Epsilon for rectangle grouping overlap.
            box_color: BGR color for bounding box drawing.
            box_thickness: Thickness of bounding box lines.
            font_scale: Font scale for label text.
            font_thickness: Font thickness for label text.
            time_font_scale: Font scale for time display.
            time_font_thickness: Font thickness for time display.
            time_bg_color: Background color for time text.
            time_text_color: Text color for time display.
        """
        # Threshold parameters
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max
        self._thresh_step = thresh_step
        
        # Area filter parameters
        self._min_area_ratio = min_area_ratio
        self._max_area_ratio = max_area_ratio
        
        # Shape parameters
        self._approx_epsilon = approx_epsilon
        self._min_aspect_ratio = min_aspect_ratio
        self._max_aspect_ratio = max_aspect_ratio
        
        # Grouping parameters
        self._group_threshold = group_threshold
        self._group_eps = group_eps
        
        # Drawing parameters
        self._box_color = box_color
        self._box_thickness = box_thickness
        self._font_scale = font_scale
        self._font_thickness = font_thickness
        self._time_font_scale = time_font_scale
        self._time_font_thickness = time_font_thickness
        self._time_bg_color = time_bg_color
        self._time_text_color = time_text_color
    
    def detect(self, image: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Detect labels in the input image.
        
        This is the main detection method that processes the image through
        multiple channels and thresholds to find consistent rectangular regions.
        
        Args:
            image: Input BGR image as numpy array.
            
        Returns:
            A tuple containing:
                - List of bounding boxes as (x, y, width, height) tuples.
                - Elapsed time in milliseconds.
        """
        start_time = time.perf_counter()
        
        if image is None:
            return [], 0.0
        
        img_height, img_width = image.shape[:2]
        img_area = img_height * img_width
        
        # Collect all candidate rectangles
        detected_candidates = []
        
        # Extract color channels
        channels = self._extract_channels(image)
        
        # Process each channel with multiple thresholds
        for channel in channels:
            for thresh_val in range(self._thresh_min, self._thresh_max, self._thresh_step):
                # Apply threshold
                binary = self._apply_threshold(channel, thresh_val)
                
                # Find contours
                contours = self._find_contours(binary)
                
                # Process each contour
                for contour in contours:
                    # Filter by area
                    if not self._filter_by_area(contour, img_area):
                        continue
                    
                    # Check if rectangle
                    if not self._is_rectangle(contour):
                        continue
                    
                    # Get bounding box and check aspect ratio
                    bbox = cv2.boundingRect(contour)
                    if self._check_aspect_ratio(bbox):
                        detected_candidates.append(bbox)
        
        # Group overlapping rectangles
        final_boxes = self._group_rectangles(detected_candidates)
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        return final_boxes, elapsed_ms
    
    def draw_results(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        elapsed_ms: float
    ) -> np.ndarray:
        """
        Draw bounding boxes and processing time on the image.
        
        Args:
            image: Input BGR image.
            boxes: List of bounding boxes as (x, y, width, height).
            elapsed_ms: Processing time in milliseconds.
            
        Returns:
            Image with drawn bounding boxes and time information.
        """
        output_img = image.copy()
        
        # Draw bounding boxes
        output_img = self._draw_bounding_boxes(output_img, boxes)
        
        # Draw time information
        output_img = self._draw_time_info(output_img, elapsed_ms)
        
        return output_img
    
    def _extract_channels(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Extract B, G, R channels from the image.
        
        Args:
            image: Input BGR image.
            
        Returns:
            List of single-channel images [B, G, R].
        """
        return cv2.split(image)
    
    def _apply_threshold(self, channel: np.ndarray, thresh_val: int) -> np.ndarray:
        """
        Apply binary threshold to a single channel.
        
        Args:
            channel: Single-channel image.
            thresh_val: Threshold value.
            
        Returns:
            Binary image.
        """
        _, binary = cv2.threshold(channel, thresh_val, 255, cv2.THRESH_BINARY)
        return binary
    
    def _find_contours(self, binary: np.ndarray) -> List[np.ndarray]:
        """
        Find external contours in a binary image.
        
        Args:
            binary: Binary image.
            
        Returns:
            List of contours.
        """
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def _filter_by_area(self, contour: np.ndarray, img_area: int) -> bool:
        """
        Check if contour area is within acceptable range.
        
        Args:
            contour: Contour to check.
            img_area: Total image area in pixels.
            
        Returns:
            True if contour area is within range, False otherwise.
        """
        area = cv2.contourArea(contour)
        min_area = img_area * self._min_area_ratio
        max_area = img_area * self._max_area_ratio
        return min_area <= area <= max_area
    
    def _is_rectangle(self, contour: np.ndarray) -> bool:
        """
        Check if contour approximates to a rectangle (4 corners).
        
        Args:
            contour: Contour to check.
            
        Returns:
            True if contour has 4 corners, False otherwise.
        """
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, self._approx_epsilon * perimeter, True)
        return len(approx) == 4
    
    def _check_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if bounding box has acceptable aspect ratio.
        
        Args:
            bbox: Bounding box as (x, y, width, height).
            
        Returns:
            True if aspect ratio is within range, False otherwise.
        """
        _, _, width, height = bbox
        if height == 0:
            return False
        aspect_ratio = float(width) / height
        return self._min_aspect_ratio < aspect_ratio < self._max_aspect_ratio
    
    def _group_rectangles(
        self,
        candidates: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Group overlapping rectangles that appear multiple times.
        
        Rectangles that appear in multiple thresholds/channels are more
        likely to be actual labels.
        
        Args:
            candidates: List of candidate bounding boxes.
            
        Returns:
            List of grouped bounding boxes.
        """
        if not candidates:
            return []
        
        # cv2.groupRectangles requires list format
        candidates_list = [list(box) for box in candidates]
        
        try:
            grouped, _ = cv2.groupRectangles(
                candidates_list,
                groupThreshold=self._group_threshold,
                eps=self._group_eps
            )
            return [tuple(box) for box in grouped]
        except cv2.error:
            # Return empty if grouping fails
            return []
    
    def _draw_bounding_boxes(
        self,
        image: np.ndarray,
        boxes: List[Tuple[int, int, int, int]]
    ) -> np.ndarray:
        """
        Draw bounding boxes with labels on the image.
        
        Args:
            image: Input image.
            boxes: List of bounding boxes.
            
        Returns:
            Image with drawn boxes.
        """
        for (x, y, w, h) in boxes:
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                self._box_color,
                self._box_thickness
            )
            cv2.putText(
                image,
                "Label Candidate",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._font_scale,
                self._box_color,
                self._font_thickness
            )
        return image
    
    def _draw_time_info(self, image: np.ndarray, elapsed_ms: float) -> np.ndarray:
        """
        Draw processing time on top-left corner of the image.
        
        Args:
            image: Input image.
            elapsed_ms: Processing time in milliseconds.
            
        Returns:
            Image with time information.
        """
        time_text = f"Time: {elapsed_ms:.2f} ms"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            time_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            self._time_font_scale,
            self._time_font_thickness
        )
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            image,
            (0, 0),
            (text_width + 2 * padding, text_height + 2 * padding + baseline),
            self._time_bg_color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image,
            time_text,
            (padding, text_height + padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._time_font_scale,
            self._time_text_color,
            self._time_font_thickness
        )
        
        return image
