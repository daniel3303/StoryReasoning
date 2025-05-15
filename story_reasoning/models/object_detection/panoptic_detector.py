import json
import os
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from tqdm import tqdm
from sklearn.cluster import KMeans

from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.utils.surpress_stdout import suppress_stdout


class PanopticDetector:
    """
    Panoptic detector class that performs both object detection and panoptic segmentation using a single model.

    This class uses the Mask2Former model to detect and segment both "thing" (object) and "stuff" (background) classes in images.
    For "stuff" classes, it employs a clustering algorithm to generate multiple bounding boxes when appropriate.

    Args:
        model_name (str): Name of the Mask2Former model to use.
        device (torch.device): Device to use for inference.
        detection_threshold (float): Detection confidence threshold.
        max_detections (int): Maximum number of detections to return per image.
    """

    def __init__(self, model_name="facebook/mask2former-swin-large-coco-panoptic", device=None,
                 detection_threshold=0.7, max_detections=40):
        self.detection_threshold = detection_threshold
        self.max_detections = max_detections
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        # Initialize the image processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)

    @staticmethod
    def load_image(image_path):
        """Load an image from a file path."""
        image = Image.open(image_path)
        return image, image.size

    def process_image(self, image):
        """Process the image through the Mask2Former model."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def post_process_segmentation(self, outputs, image_size):
        """Post-process the model outputs to get panoptic segmentation results."""
        return self.processor.post_process_panoptic_segmentation(
            outputs,
            target_sizes=[image_size[::-1]],
            threshold=self.detection_threshold,
        )[0]

    @staticmethod
    def normalize_coordinates(box, image_width, image_height):
        """Normalize bounding box coordinates to be in the range [0, 1]."""
        x_min, y_min, x_max, y_max = box
        return {
            "x": x_min / image_width,
            "y": y_min / image_height,
            "w": (x_max - x_min) / image_width,
            "h": (y_max - y_min) / image_height
        }

    @staticmethod
    def sort_detections(detections):
        """Sort detections based on their center position in the image.

        The image is divided into 3 horizontal bands. Objects are sorted:
        1. First by the band they belong to (top to bottom)
        2. Within each band, from left to right based on center point
        3. If the band and x-coordinates are the same, from top to bottom based on center point

        Args:
            detections (List[Detection]): List of Detection objects to sort

        Returns:
            List[Detection]: Sorted list of detections
        """

        def get_center_coords(detection):
            # Calculate center point of the bounding box
            center_x = detection.box_x + (detection.box_w / 2)
            center_y = detection.box_y + (detection.box_h / 2)

            n_bands = 3

            # Determine which third of the image the center falls into (0, 1, ..., n_bands - 1)
            band = int(center_y * n_bands)

            # Clamp to valid range in case of floating point imprecision
            band = max(0, min(n_bands - 1, band))

            return (band, center_x, center_y)

        return sorted(detections, key=get_center_coords)


    @staticmethod
    def assign_ids(detections):
        """Assign unique IDs to detections of the same class."""
        label_id_counters = {}
        for detection in detections:
            if detection.label not in label_id_counters:
                label_id_counters[detection.label] = 0
            detection.id = label_id_counters[detection.label]
            label_id_counters[detection.label] += 1
        return detections

    @staticmethod
    def clean_label(label):
        """Clean up label names by removing certain suffixes."""
        return label.replace('-other-merged', '').replace('-merged', '').replace('-other', '')

    @staticmethod
    def calculate_edge(points, ignore_ratio, func):
        """
        Calculate an edge of a bounding box based on a list of points.

        Args:
            points: List of points (single coordinate) to calculate the edge.
            ignore_ratio: Ratio of points to ignore from the start and end of the sorted list.
            func: Function to use for calculating the edge (np.min or np.max).

        Returns:
            The calculated edge value.
        """
        if len(points) < 2:
            return None
        sorted_points = np.sort(points)
        ignore_index = int(len(sorted_points) * ignore_ratio)
        if func == np.min:
            return sorted_points[ignore_index].item()
        elif func == np.max:
            return sorted_points[-ignore_index - 1].item() if ignore_index > 0 else sorted_points[-1].item()

    @staticmethod
    def generate_bounding_boxes(mask, max_clusters=6, min_coverage=0.9, max_overflow=0.3, n_initializations=6):
        """
        Generate bounding boxes for objects in a binary mask image using K-means clustering.

        This method attempts to find the optimal number of clusters (boxes) that provide
        good coverage of the masked area while minimizing overflow.

        Parameters:
            mask (numpy.ndarray): A binary mask where non-zero pixels represent the objects of interest.
            max_clusters (int, optional): The maximum number of clusters to try. Default is 5.
            min_coverage (float, optional): The minimum fraction of masked pixels that should be
                                            covered by the bounding boxes. Default is 0.9.
            max_overflow (float, optional): The maximum fraction of non-masked pixels that can be
                                            included in the bounding boxes. Default is 0.3.
            n_initializations (int, optional): The number of times to run K-means with different
                                               initializations for each cluster count. Default is 4.

        Returns:
            List[List[float]]: A list of bounding boxes, where each box is represented as
                               [x_min, y_min, x_max, y_max] in normalized coordinates (0 to 1).

        Algorithm:
        1. Extract the coordinates of non-zero pixels in the mask.
        2. Iterate through possible numbers of clusters from 1 to max_clusters:
           a. For each cluster count, perform K-means clustering n_initializations times.
           b. For each initialization:
              - Generate initial bounding boxes based on the clustered points.
              - Adjust overlapping boxes to eliminate overlaps.
              - Calculate coverage and overflow ratios for the adjusted boxes.
              - Update the best result for this cluster count if the current result is better.
           c. Update the global best result if the best result for this cluster count is better than the previous best.
           d. If the best result for this cluster count meets the minimum coverage and maximum overflow criteria, stop iterating.
        3. Return the best set of bounding boxes found.

        Notes:
        - The method uses a custom algorithm to generate initial bounding boxes that attempts to
          include a specified percentage of points while excluding extreme outliers.
        - Overlapping boxes are adjusted using an algorithm that minimizes distortion while
          completely resolving overlaps.
        - The method balances between maximizing coverage of the masked area and minimizing
          overflow into non-masked areas.
        - Multiple initializations for each cluster count help to avoid local optima in the
          K-means clustering.
        """
        y, x = np.where(mask)
        if len(x) == 0 or len(y) == 0:
            return []

        points = np.column_stack((x, y))
        total_mask_area = np.sum(mask)

        best_global_boxes = []
        best_global_score = 0

        for n_clusters in range(1, max_clusters + 1):
            best_cluster_boxes = []
            best_cluster_score = 0
            best_cluster_coverage = 0
            best_cluster_overflow = 0

            for _ in range(n_initializations):
                kmeans = KMeans(n_clusters=n_clusters, max_iter=500)
                kmeans.fit(points)

                # Generate initial bounding boxes
                """
                Algorithm for generating initial bounding boxes:

                1. For each cluster identified by KMeans:
                   a. Extract all points belonging to the cluster.
                   b. Calculate the center of the cluster using the mean of all points.
                   c. Divide the points into four groups based on their position relative to the center:
                      - Top points: y < center_y
                      - Bottom points: y >= center_y
                      - Left points: x < center_x
                      - Right points: x >= center_x
                   d. For each edge (top, bottom, left, right):
                      - Calculate the ignore_ratio as (1 - min_coverage) / 2. This represents the proportion
                        of points to ignore at each extreme, ensuring symmetrical coverage.
                      - Use calculate_edge function to determine the edge position:
                        * For top and left: find the minimum value after ignoring the lowest ignore_ratio of points.
                        * For bottom and right: find the maximum value after ignoring the highest ignore_ratio of points.
                   e. If any edge couldn't be calculated (due to insufficient points), skip this cluster.
                   f. Adjust the edges to ensure the box is within the image boundaries.
                   g. Add the resulting box to the list of initial boxes.

                2. The calculate_edge function:
                   - Sorts the relevant points.
                   - Determines the index based on the ignore_ratio.
                   - Returns the appropriate value (min or max) from the sorted points, excluding the ignored portion.

                This approach ensures that:
                - Each box is centered on the cluster's center of mass.
                - A specified percentage (min_coverage) of points is included in each dimension.
                - Extreme outliers are excluded symmetrically from both sides.
                - Boxes are always within the image boundaries.
                - Clusters with too few points to form a valid box are ignored.
                """
                initial_boxes = []
                for cluster in range(n_clusters):
                    cluster_points = points[kmeans.labels_ == cluster]
                    if len(cluster_points) == 0:
                        continue

                    center_x, center_y = np.mean(cluster_points, axis=0)

                    top_points = cluster_points[cluster_points[:, 1] < center_y]
                    bottom_points = cluster_points[cluster_points[:, 1] >= center_y]
                    left_points = cluster_points[cluster_points[:, 0] < center_x]
                    right_points = cluster_points[cluster_points[:, 0] >= center_x]

                    ignore_ratio = (1 - min_coverage) / 2
                    top = PanopticDetector.calculate_edge(top_points[:, 1], ignore_ratio, np.min)
                    bottom = PanopticDetector.calculate_edge(bottom_points[:, 1], ignore_ratio, np.max)
                    left = PanopticDetector.calculate_edge(left_points[:, 0], ignore_ratio, np.min)
                    right = PanopticDetector.calculate_edge(right_points[:, 0], ignore_ratio, np.max)

                    if top is None or bottom is None or left is None or right is None:
                        continue

                    top = max(0, top)
                    bottom = min(mask.shape[0] - 1, bottom)
                    left = max(0, left)
                    right = min(mask.shape[1] - 1, right)

                    initial_boxes.append([left, top, right, bottom])

                # Adjust overlapping boxes
                """
                Algorithm for adjusting overlapping bounding boxes:

                1. Iterate through all pairs of bounding boxes.
                2. For each pair, check if they overlap in both x and y directions.
                3. If overlap exists:
                   a. Calculate the overlap percentage relative to the total size of both boxes
                      in each direction (x and y).
                   b. Choose the direction (x or y) with the smaller relative overlap percentage
                      for adjustment. This minimizes the overall change to the boxes.
                   c. Calculate how much each box should move based on its size relative to 
                      the total size of both boxes in the chosen direction. Larger boxes will
                      move more than smaller boxes.
                   d. Adjust the boxes in the chosen direction, moving them apart just enough
                      to eliminate the overlap.
                4. After adjustments, ensure all box coordinates remain valid.

                This approach ensures minimal distortion to the original bounding boxes while
                completely resolving overlaps. It takes into account the relative sizes of the
                boxes, allowing for more balanced and size-appropriate adjustments.
                """
                for i in range(len(initial_boxes)):
                    for j in range(i + 1, len(initial_boxes)):
                        box1 = initial_boxes[i]
                        box2 = initial_boxes[j]

                        x_overlap = min(box1[2], box2[2]) - max(box1[0], box2[0])
                        y_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])

                        if x_overlap > 0 and y_overlap > 0:
                            width1, height1 = box1[2] - box1[0], box1[3] - box1[1]
                            width2, height2 = box2[2] - box2[0], box2[3] - box2[1]
                            total_width = width1 + width2
                            total_height = height1 + height2

                            x_overlap_percent = x_overlap / total_width
                            y_overlap_percent = y_overlap / total_height

                            if x_overlap_percent < y_overlap_percent:
                                adjustment = x_overlap
                                ratio1 = width1 / total_width
                                ratio2 = width2 / total_width

                                if box1[0] < box2[0]:
                                    box1[2] -= adjustment * ratio1
                                    box2[0] += adjustment * ratio2
                                else:
                                    box1[0] += adjustment * ratio1
                                    box2[2] -= adjustment * ratio2
                            else:
                                adjustment = y_overlap
                                ratio1 = height1 / total_height
                                ratio2 = height2 / total_height

                                if box1[1] < box2[1]:
                                    box1[3] -= adjustment * ratio1
                                    box2[1] += adjustment * ratio2
                                else:
                                    box1[1] += adjustment * ratio1
                                    box2[3] -= adjustment * ratio2

                        for box in [box1, box2]:
                            box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
                            box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])

                # Calculate coverage and overflow for adjusted boxes
                current_boxes = []
                total_coverage = 0
                total_overflow = 0

                for box in initial_boxes:
                    x_min, y_min, x_max, y_max = map(int, box)
                    box_mask = np.zeros_like(mask)
                    box_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

                    coverage = np.sum(mask & box_mask)
                    overflow = np.sum(box_mask) - coverage

                    total_coverage += coverage
                    total_overflow += overflow

                    current_boxes.append(
                        PanopticDetector.normalize_coordinates([x_min, y_min, x_max, y_max], mask.shape[1],
                                                               mask.shape[0])
                    )

                current_coverage = total_coverage / total_mask_area
                current_overflow = total_overflow / total_mask_area
                current_score = current_coverage - current_overflow

                if current_score > best_cluster_score:
                    best_cluster_boxes = current_boxes
                    best_cluster_score = current_score
                    best_cluster_coverage = current_coverage
                    best_cluster_overflow = current_overflow

                if best_cluster_coverage >= min_coverage and best_cluster_overflow <= max_overflow:
                    break

            if best_cluster_score > best_global_score:
                best_global_boxes = best_cluster_boxes
                best_global_score = best_cluster_score

            if best_cluster_coverage >= min_coverage and best_cluster_overflow <= max_overflow:
                break

        return best_global_boxes

    def get_detections(self, segmentation_result, image_size):
        """
        Extract detections from the segmentation result.

        This method processes both "thing" (object) and "stuff" (background) classes.
        For "stuff" classes, it uses the generate_bounding_boxes method to create multiple bounding boxes when appropriate.

        Args:
            segmentation_result (dict): The result of the panoptic segmentation.
            image_size (tuple): The size of the input image (width, height).

        Returns:
            List[Detection]: A list of Detection objects representing all detected instances.
        """
        panoptic_seg = segmentation_result["segmentation"]
        segments_info = segmentation_result["segments_info"]
        image_w, image_h = image_size


        detections = []
        for segment_info in segments_info:
            if segment_info["score"] > self.detection_threshold:
                mask = panoptic_seg == segment_info["id"]
                label = self.clean_label(self.model.config.id2label[segment_info["label_id"]])

                # Convert mask to numpy and get normalized version
                mask_np = mask.cpu().numpy()
                y, x = np.where(mask_np)
                if len(x) == 0 or len(y) == 0:
                    continue

                x_min, x_max = np.min(x), np.max(x)
                y_min, y_max = np.min(y), np.max(y)

                # Create normalized mask matching the detection's bounding box
                normalized_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=bool)
                normalized_mask[y[y >= y_min] - y_min, x[x >= x_min] - x_min] = True

                if segment_info["label_id"] > 90:  # Stuff classes
                    bounding_boxes = self.generate_bounding_boxes(mask_np)
                    for box in bounding_boxes:
                        # Get mask for this specific box
                        box_x1 = int(box['x'] * image_w)
                        box_y1 = int(box['y'] * image_h)
                        box_x2 = int((box['x'] + box['w']) * image_w)
                        box_y2 = int((box['y'] + box['h']) * image_h)

                        box_mask = mask_np[box_y1:box_y2, box_x1:box_x2]

                        detection = Detection(
                            id=0,  # Temporary ID, will be updated later
                            label=label,
                            image_id=None,
                            score=segment_info["score"],
                            box_x=box['x'],
                            box_y=box['y'],
                            box_w=box['w'],
                            box_h=box['h'],
                            image_width=image_w,
                            image_height=image_h,
                            mask=box_mask,
                            is_landmark=False,
                            is_thing=False
                        )
                        detections.append(detection)
                else:  # Object classes
                    normalized_box = self.normalize_coordinates(
                        [x_min, y_min, x_max, y_max], image_w, image_h
                    )

                    detection = Detection(
                        id=0,  # Temporary ID, will be updated later
                        label=label,
                        image_id=None,
                        score=segment_info["score"],
                        box_x=normalized_box['x'],
                        box_y=normalized_box['y'],
                        box_w=normalized_box['w'],
                        box_h=normalized_box['h'],
                        image_width=image_w,
                        image_height=image_h,
                        mask=normalized_mask,
                        is_landmark=False,
                        is_thing=True
                    )
                    detections.append(detection)

        sorted_detections = self.sort_detections(detections)
        assigned_detections = self.assign_ids(sorted_detections)
        return assigned_detections[:self.max_detections]

    @staticmethod
    def draw_bounding_boxes(image, detections, use_id_as_label:bool = True, show_score:bool = False) -> Image:
        """
        Draw bounding boxes and labels on the input image.

        Args:
            image (PIL.Image.Image or str): The input image or path to the image.
            detections (List[Detection]): The list of detections to draw.
            use_id_as_label (bool): Whether to use detection IDs as labels. Default is True.
            show_score (bool): Whether to show detection scores in labels. Default is False.

        Returns:
            PIL.Image.Image: The image with bounding boxes and labels drawn on it.
        """
        if isinstance(image, str):
            image = Image.open(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        all_colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta", "grey"]
        available_colors = all_colors.copy()
        color_per_label = {}

        for detection in detections:
            x_min = int(detection.box_x * image.width)
            y_min = int(detection.box_y * image.height)
            x_max = int((detection.box_x + detection.box_w) * image.width)
            y_max = int((detection.box_y + detection.box_h) * image.height)

            if detection.label in color_per_label:
                color = color_per_label[detection.label]
            else:
                if len(available_colors) == 0:
                    available_colors = all_colors.copy()
                color = available_colors.pop(0)
                color_per_label[detection.label] = color

            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)
            if use_id_as_label:
                draw.text((x_min, y_min), f"{detection.get_global_id()}" + (f": {detection.score:.2f}" if show_score else ""), fill=color, font=font)
            else:
                draw.text((x_min, y_min), f"{detection.label}" + (f": {detection.score:.2f}" if show_score else ""), fill=color, font=font)

        return image

    def analyze(self, image) -> List[Detection]:
        """
        Analyze an image and return a list of detections.

        This method performs the following steps:
        1. Load the image
        2. Process the image through the Mask2Former model
        3. Post-process the segmentation results
        4. Extract detections from the segmentation results

        Args:
            image (PIL.Image.Image or str): The input image or path to the image.

        Returns:
            List[Detection]: A list of Detection objects representing all detected instances in the image
                these detections have unique ids and are sorted based on their position in the image.
        """
        if isinstance(image, str):
            image, image_size = self.load_image(image)
        else:
            image_size = image.size

        outputs = self.process_image(image)
        segmentation_result = self.post_process_segmentation(outputs, image_size)
        detections = self.get_detections(segmentation_result, image_size)
        return detections


    def process_folder(self, input_folder, output_folder, max_images=None):
        """
        Process all images in a folder and save the detection results.

        This method performs the following steps for each image in the input folder:
        1. Analyze the image using the 'analyze' method
        2. Save the detection results as a JSON file in the output folder
        3. Keep track of the results for each image

        Args:
            input_folder (str): Path to the folder containing input images.
            output_folder (str): Path to the folder where output JSON files will be saved.
            max_images (int, optional): Maximum number of images to process. If None, process all images.

        Returns:
            List[dict]: A list of dictionaries containing information about each processed image.
        """
        os.makedirs(output_folder, exist_ok=True)

        image_files = sorted(
            [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

        if max_images is not None:
            image_files = image_files[:max_images]

        results = []

        for image_file in tqdm(image_files, desc="Processing images"):
            input_path = os.path.join(input_folder, image_file)

            detections = self.analyze(input_path)

            output_filename = os.path.splitext(image_file)[0] + '.json'
            output_path = os.path.join(output_folder, output_filename)

            with open(output_path, 'w') as f:
                json.dump([d.to_dict() for d in detections], f, indent=2)

            results.append({
                'input_file': image_file,
                'output_file': output_filename,
                'num_detections': len(detections)
            })

        return results
