import argparse

from story_reasoning.models.object_detection.panoptic_detector import PanopticDetector


def main():
    """
    Main function for the image analysis command-line interface.

    This function sets up the argument parser, initializes the ImageAnalyzer,
    processes the specified folder of images, and prints a summary of the results.

    Command-line Arguments:
        input_folder (str): Path to the folder containing input images.
        output_folder (str): Path to the folder where JSON results will be saved.
        --max_images (int, optional): Maximum number of images to process.
            If not specified, all images in the input folder will be processed.
        --object_detection_model (str, optional): Name of the object detection model to use.
            Default is "facebook/detr-resnet-101".
        --panoptic_model (str, optional): Name of the panoptic segmentation model to use.
            Default is "facebook/mask2former-swin-base-coco-panoptic".

    Usage:
        python script_name.py input_folder output_folder [--max_images MAX_IMAGES]
                              [--object_detection_model MODEL_NAME]
                              [--panoptic_model MODEL_NAME]

    Examples:
        Process all images in a folder:
        python script_name.py /path/to/input/folder /path/to/output/folder

        Process a maximum of 5 images:
        python script_name.py /path/to/input/folder /path/to/output/folder --max_images 5

        Use different models:
        python script_name.py /path/to/input/folder /path/to/output/folder
                              --object_detection_model "facebook/detr-resnet-101"
                              --panoptic_model "facebook/mask2former-swin-large-coco-panoptic"

    Returns:
        None. Results are saved as JSON files in the specified output folder.
    """

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Analyze images and detect objects.")
    parser.add_argument("input_folder", help="Path to the folder containing input images")
    parser.add_argument("output_folder", help="Path to the folder where JSON results will be saved")
    parser.add_argument("--max_images", type=int, help="Maximum number of images to process (default: process all)", default=None)
    parser.add_argument("--object_detection_model", default="facebook/detr-resnet-101", help="Name of the object detection model to use")
    parser.add_argument("--panoptic_model", default="facebook/mask2former-swin-base-coco-panoptic", help="Name of the panoptic segmentation model to use")

    # Parse config
    args = parser.parse_args()

    # Initialize the ImageAnalyzer
    analyzer = PanopticDetector()

    # Process the folder
    results = analyzer.process_folder(args.input_folder, args.output_folder, max_images=args.max_images)

    # Print summary of processed images
    print(f"\nProcessed {len(results)} images:")
    for result in results:
        print(f"  {result['input_file']} -> {result['output_file']} ({result['num_detections']} detections)")

if __name__ == "__main__":
    main()