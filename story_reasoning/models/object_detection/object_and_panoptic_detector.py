import json
import os
import warnings
from collections import defaultdict
from typing import List, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Mask2FormerForUniversalSegmentation
from transformers import logging

from story_reasoning.models.object_detection.detection import Detection

logging.set_verbosity_error()

# Deprecated: Use PanopticDetector instead.
@DeprecationWarning
class ObjectAndPanopticDetector:
    """
    Object and panoptic detector class that combines object detection and panoptic segmentation models.
    Deprecated: Use PanopticDetector instead.

    Args:
        object_detection_model_name (str): Name of the object detection model.
        panoptic_model_name (str): Name of the panoptic segmentation model.
        device (torch.device): Device to use for inference.
        object_detection_model_args (dict): Additional config for the object detection model.
        panoptic_model_args (dict): Additional config for the panoptic segmentation model.
        object_threshold (float): Detection threshold.
        panoptic_threshold (float): Detection threshold for panoptic segmentation.
        max_objects (int): Maximum number of object detections to keep.
        max_objects_per_class (int): Maximum number of object detections per class to keep.
        max_panoptic_segments (int): Maximum number of panoptic segments to keep.
    """

    def __init__(self, object_detection_model_name = "facebook/detr-resnet-101", panoptic_model_name = "facebook/mask2former-swin-large-coco-panoptic",
                 device=None, object_detection_model_args=None, panoptic_model_args=None, object_threshold=0.7,
                 panoptic_threshold=0.7, max_objects=60,max_objects_per_class=25, max_panoptic_segments=20):
        warnings.warn(
            "ObjectAndPanopticDetector is deprecated. Use PanopticDetector instead.",
            DeprecationWarning
        )
        self.object_threshold = object_threshold
        self.panoptic_threshold = panoptic_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.max_objects = max_objects
        self.max_objects_per_class = max_objects_per_class
        self.max_panoptic_segments = max_panoptic_segments

        self.object_detection_processor = AutoImageProcessor.from_pretrained(object_detection_model_name,
                                                                             **(object_detection_model_args or {}))
        self.object_detection_model = AutoModelForObjectDetection.from_pretrained(object_detection_model_name, **(
                object_detection_model_args or {})).to(self.device)

        self.panoptic_processor = AutoImageProcessor.from_pretrained(panoptic_model_name, **(panoptic_model_args or {}))
        self.panoptic_model = Mask2FormerForUniversalSegmentation.from_pretrained(panoptic_model_name,
                                                                                  **(panoptic_model_args or {})).to(self.device)

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path)
        return image, image.size

    def process_image(self, image):
        object_detection_inputs = self.object_detection_processor(images=image, return_tensors="pt").to(self.device)
        panoptic_inputs = self.panoptic_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            object_detection_outputs = self.object_detection_model(**object_detection_inputs)
            panoptic_outputs = self.panoptic_model(**panoptic_inputs)

        return object_detection_outputs, panoptic_outputs

    def post_process_object_detection(self, outputs, image_size):
        target_sizes = torch.tensor([image_size[::-1]]).to(self.device)
        return self.object_detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.object_threshold)[0]

    def post_process_panoptic_segmentation(self, outputs, image_size):
        return self.panoptic_processor.post_process_panoptic_segmentation(
            outputs, overlap_mask_area_threshold=0.8, threshold=self.panoptic_threshold,
            target_sizes=[image_size[::-1]])[0]

    @staticmethod
    def normalize_coordinates(box, image_width, image_height):
        x_min, y_min, x_max, y_max = box
        return {
            "x": x_min / image_width,
            "y": y_min / image_height,
            "w": (x_max - x_min) / image_width,
            "h": (y_max - y_min) / image_height
        }

    @staticmethod
    def sort_detections(detections):
        return sorted(detections, key=lambda d: (d.box_y / 0.2, d.box_x, d.box_y))

    @staticmethod
    def assign_ids(detections):
        label_id_counters = {}
        for detection in detections:
            if detection.label not in label_id_counters:
                label_id_counters[detection.label] = 0
            detection.id = label_id_counters[detection.label]
            label_id_counters[detection.label] += 1
        return detections

    def get_object_detections(self, results, image_size):
        object_detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i) for i in box.tolist()]
            class_label = self.object_detection_model.config.id2label[label.item()]
            normalized_box = self.normalize_coordinates(box, *image_size)
            detection = Detection(
                id=0,  # Temporary ID, will be updated later
                label=class_label,
                image_id=None,
                score=score.item(),
                box_x=normalized_box['x'],
                box_y=normalized_box['y'],
                box_w=normalized_box['w'],
                box_h=normalized_box['h'],
                image_width=image_size[0],
                image_height=image_size[1],
                is_landmark=False,
                is_thing=True,
            )
            object_detections.append(detection)

        sorted_detections = self.sort_detections(object_detections)
        return self.assign_ids(sorted_detections)

    def filter_object_detections(self, object_detections):
        object_detections.sort(key=lambda x: x.score, reverse=True)
        filtered_detections = []
        class_count = defaultdict(int)
        for detection in object_detections:
            if class_count[detection.label] < self.max_objects_per_class:
                filtered_detections.append(detection)
                class_count[detection.label] += 1
            if len(filtered_detections) == self.max_objects:
                break
        return filtered_detections

    @staticmethod
    def calculate_edge(points, ignore_ratio, func) -> Union[int, None]:
        """
            Calculate an edge of a bounding box based on a list of points.
            The points should include only the coordinate of the point on the intended edge.
        Args:
            points: List of points (single coordinate) to calculate the edge
            ignore_ratio: Ratio of points to ignore from the start and end of the sorted list.
            func: Function to use for calculating the edge (np.min or np.max)

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
    def generate_bounding_boxes(mask, max_clusters=6, min_coverage=0.9, max_overflow=0.3, n_initializations = 4):
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

                # Initial box calculation
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

                    # Calculate center
                    center_x, center_y = np.mean(cluster_points, axis=0)

                    # Separate points for each edge calculation
                    top_points = cluster_points[cluster_points[:, 1] < center_y]
                    bottom_points = cluster_points[cluster_points[:, 1] >= center_y]
                    left_points = cluster_points[cluster_points[:, 0] < center_x]
                    right_points = cluster_points[cluster_points[:, 0] >= center_x]

                    # Calculate edges
                    ignore_ratio = (1 - min_coverage) / 2
                    top = ObjectAndPanopticDetector.calculate_edge(top_points[:, 1], ignore_ratio, np.min)
                    bottom = ObjectAndPanopticDetector.calculate_edge(bottom_points[:, 1], ignore_ratio, np.max)
                    left = ObjectAndPanopticDetector.calculate_edge(left_points[:, 0], ignore_ratio, np.min)
                    right = ObjectAndPanopticDetector.calculate_edge(right_points[:, 0], ignore_ratio, np.max)

                    # Ignore boxes that are too small
                    if top is None or bottom is None or left is None or right is None:
                        continue

                    # Ensure box is within image boundaries
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

                        # Check for overlap
                        x_overlap = min(box1[2], box2[2]) - max(box1[0], box2[0])
                        y_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])

                        if x_overlap > 0 and y_overlap > 0:
                            # Boxes overlap in both dimensions. We need to adjust them.

                            # Calculate box dimensions
                            width1, height1 = box1[2] - box1[0], box1[3] - box1[1]
                            width2, height2 = box2[2] - box2[0], box2[3] - box2[1]

                            # Calculate total dimensions (sum of both boxes)
                            total_width = width1 + width2
                            total_height = height1 + height2

                            # Calculate overlap percentages relative to total dimensions
                            x_overlap_percent = x_overlap / total_width
                            y_overlap_percent = y_overlap / total_height

                            if x_overlap_percent < y_overlap_percent:
                                # Adjust in x direction as it has the smaller relative overlap
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
                                # Adjust in y direction as it has the smaller relative overlap
                                adjustment = y_overlap
                                ratio1 = height1 / total_height
                                ratio2 = height2 / total_height

                                if box1[1] < box2[1]:
                                    box1[3] -= adjustment * ratio1
                                    box2[1] += adjustment * ratio2
                                else:
                                    box1[1] += adjustment * ratio1
                                    box2[3] -= adjustment * ratio2

                        # After adjustments, ensure box coordinates are still valid
                        for box in [box1, box2]:
                            box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
                            box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])

                # Calculate coverage and overflow for adjusted boxes
                current_boxes = []
                total_coverage = 0
                total_overflow = 0

                for box in initial_boxes:
                    x_min, y_min, x_max, y_max = map(int, box)  # Ensure integer coordinates
                    box_mask = np.zeros_like(mask)
                    box_mask[y_min:y_max + 1, x_min:x_max + 1] = 1

                    coverage = np.sum(mask & box_mask)
                    overflow = np.sum(box_mask) - coverage

                    # we can sum the coverage and overflow of all boxes because they are non-overlapping
                    total_coverage += coverage
                    total_overflow += overflow

                    current_boxes.append(
                        ObjectAndPanopticDetector.normalize_coordinates([x_min, y_min, x_max, y_max], mask.shape[1],
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

            # Update best overall boxes if this clustering is better
            if best_cluster_score > best_global_score:
                best_global_boxes = best_cluster_boxes
                best_global_score = best_cluster_score

            if best_cluster_coverage >= min_coverage and best_cluster_overflow <= max_overflow:
                break

        return best_global_boxes

    @staticmethod
    def clean_label(label):
        return label.replace('-other-merged', '').replace('-merged', '').replace('-other', '').replace('-stuff', '')

    def get_panoptic_detections(self, panoptic_result):
        panoptic_seg = panoptic_result["segmentation"]
        segments_info = panoptic_result["segments_info"]

        panoptic_detections = []
        for segment_info in segments_info:
            if segment_info["score"] > 0.5 and segment_info["label_id"] > 90:  # Only consider "stuff" classes
                mask = panoptic_seg == segment_info["id"]
                mask_np = mask.cpu().numpy()

                bounding_boxes = self.generate_bounding_boxes(mask_np)
                label = self.clean_label(self.panoptic_model.config.id2label[segment_info["label_id"]])

                for box in bounding_boxes:
                    detection = Detection(
                        id=0,  # Temporary ID, will be updated later
                        label=label,
                        image_id=None,
                        score=segment_info.get("score", 1.0),
                        box_x=box['x'],
                        box_y=box['y'],
                        box_w=box['w'],
                        box_h=box['h'],
                        image_width= mask_np.shape[1],
                        image_height= mask_np.shape[0],
                        is_landmark=False,
                        is_thing=False,
                    )
                    panoptic_detections.append(detection)

        sorted_detections = self.sort_detections(panoptic_detections)
        assigned_detections = self.assign_ids(sorted_detections)
        assigned_detections.sort(key=lambda x: x.box_w * x.box_h, reverse=True)
        return assigned_detections[:self.max_panoptic_segments]

    @staticmethod
    def combine_detections(object_detections, panoptic_detections):
        combined_detections = object_detections + panoptic_detections
        combined_detections.sort(key=lambda x: x.score, reverse=True)
        return combined_detections

    @staticmethod
    def draw_bounding_boxes(image, combined_detections) -> Image:
        if isinstance(image, str):
            image = Image.open(image)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        all_colors = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "brown", "cyan", "magenta", "grey"]
        available_colors = all_colors.copy()
        color_per_label = {}
        for detection in combined_detections:
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
            draw.text((x_min, y_min), f"{detection.label}: {detection.score:.2f}", fill=color, font=font)

        return image

    def analyze(self, image_path) -> List[Detection]:
        image, image_size = self.load_image(image_path)
        object_detection_outputs, panoptic_outputs = self.process_image(image)

        object_detection_results = self.post_process_object_detection(object_detection_outputs, image_size)
        object_detections = self.get_object_detections(object_detection_results, image_size)
        filtered_object_detections = self.filter_object_detections(object_detections)

        panoptic_result = self.post_process_panoptic_segmentation(panoptic_outputs, image_size)
        panoptic_detections = self.get_panoptic_detections(panoptic_result)

        combined_detections = filtered_object_detections + panoptic_detections
        return self.sort_detections(combined_detections)

    def process_folder(self, input_folder, output_folder, max_images=None):
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
