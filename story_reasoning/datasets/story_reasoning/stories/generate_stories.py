import argparse
import base64
import gc
import io
import multiprocessing
import os
import random
import re
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from PIL import Image
from httpx import Timeout
from openai import OpenAI

from story_reasoning.datasets.story_reasoning.messages.request_message import RequestMessage
from story_reasoning.datasets.story_reasoning.story_reasoning_derived_dataset import StoryReasoningDerivedDataset
from story_reasoning.models.landmark_detection.google_landmark_detector import GoogleLandmarkDetector
from story_reasoning.models.landmark_detection.llm_landmark_detector import LLMLandmarkDetector
from story_reasoning.models.object_detection.detection import Detection
from story_reasoning.models.object_detection.panoptic_detector import PanopticDetector
from story_reasoning.models.object_matching.siglip_matcher import SigLipMatcher
from story_reasoning.models.story_reasoning.story_reasoning_util import StoryReasoningUtil
from story_reasoning.models.story_reasoning.narrative_phase import NarrativePhase
from story_reasoning.models.story_reasoning.setting import SettingElement
from story_reasoning.models.story_reasoning.cot import CoT
from story_reasoning.utils.surpress_stdout import suppress_stdout


def free_memory():
    if torch.cuda.is_available():
        # First, synchronize to ensure all GPU operations have completed
        torch.cuda.synchronize()
        
        # Then, empty the cache to free up memory
        torch.cuda.empty_cache()
    
    # Also force the Garbage Collector to run
    gc.collect()

    
    
def detect_and_match_objects(images: List[Image.Image]) -> Tuple[List[List[Detection]], Dict[str, str]]:
    """
    Detect objects in all images and match them across images.
    
    Args:
        images: List of PIL Image objects
        
    Returns:
        Tuple containing:
            - List of detection lists for each image
            - Dictionary mapping global object IDs to consistent IDs
    """
    

    # Analyze the images directly, suppressing stdout for a cleaner output
    all_detections = []
    with suppress_stdout():
        detector = PanopticDetector()
        for idx, img in enumerate(images):
            # Pass the image directly to analyze
            detections = detector.analyze(img)
    
            # Add image_id to each detection
            for det in detections:
                det.image_id = str(idx)  # Convert idx to string
    
            all_detections.append(detections)

    # For memory usage improvement, delete the detector
    del detector
    free_memory()


    # Match objects across images
    print("Matching objects across images...")
    matcher = SigLipMatcher()
    matches = matcher.match_detections(all_detections, images)
    
    # For memory usage improvement, delete the matcher
    free_memory()

    # Create a mapping of object IDs to consistent IDs
    object_id_map = {}

    for match_idx, match in enumerate(matches):
        # Ignore matches with only one detection if they are not things
        if len(match.detections) == 1 and not match.detections[0].is_thing and not match.detections[0].is_landmark:
            continue

        # Map each detection's ID to the match's ID
        for det in match.detections:
            object_id_map[det.get_global_id()] = match.id

    return all_detections, object_id_map


def detect_landmarks(images: List[Image.Image], google_credentials_path: str = None, llm_base_url=None,
                     llm_model: str = None, llm_api_key: str = None) -> List[List[Detection]]:
    """
    Detect landmarks in all images using Google Cloud Vision and/or LLM detector.
    
    Args:
        images: List of PIL Image objects
        google_credentials_path: Path to Google Cloud credentials JSON file
        llm_base_url: Base URL for LLM detector
        llm_model: Model name for LLM detector
        llm_api_key: API key for the LLM landmark detector
        
    Returns:
        List of landmark detection lists for each image
    """
    # Initialize landmark detectors
    google_detector = None
    llm_detector = None

    if google_credentials_path and os.path.exists(google_credentials_path):
        try:
            google_detector = GoogleLandmarkDetector(credentials_path=google_credentials_path)
            print("Google Landmark Detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Google Landmark Detector: {e}")

    if llm_api_key:
        try:
            with suppress_stdout():
                llm_detector = LLMLandmarkDetector(
                    base_url=llm_base_url,
                    model_name=llm_model,
                    api_key=llm_api_key
                )
            print("LLM Landmark Detector initialized successfully")
        except Exception as e:
            print(f"Failed to initialize LLM Landmark Detector: {e}")

    if not google_detector and not llm_detector:
        print("Warning: No landmark detectors available. Skipping landmark detection.")
        return [[] for _ in images]

    all_landmarks = []

    for idx, img in enumerate(images):
        try:
            image_landmarks = []

            # Try Google detector first
            if google_detector:
                google_landmarks = google_detector.analyze(img)
                if google_landmarks:
                    # Mark detections as landmarks
                    for det in google_landmarks:
                        det.image_id = str(idx)
                        det.is_landmark = True
                    image_landmarks = google_landmarks

            # If no Google results and LLM detector is available, try LLM detector
            if not image_landmarks and llm_detector:
                llm_landmarks = llm_detector.analyze(img)
                # Mark detections as landmarks
                for det in llm_landmarks:
                    det.image_id = str(idx)
                    det.is_landmark = True
                image_landmarks = llm_landmarks

            all_landmarks.append(image_landmarks)

        except Exception as e:
            print(f"Error detecting landmarks in image {idx}: {e}")
            all_landmarks.append([])

    return all_landmarks


def merge_detections(object_detections: List[List[Detection]],
                     landmark_detections: List[List[Detection]]) -> List[List[Detection]]:
    """
    Merge object detections and landmark detections into a single list for each image.
    
    Args:
        object_detections: List of object detection lists for each image
        landmark_detections: List of landmark detection lists for each image
        
    Returns:
        List of merged detection lists for each image
    """
    merged_detections = []

    for i in range(len(object_detections)):
        image_objects = object_detections[i]
        image_landmarks = landmark_detections[i]

        # Combine detections
        merged = image_objects + image_landmarks
        merged_detections.append(merged)

    return merged_detections


def generate_cot(client, images: List[Image.Image], all_detections: List[List[Detection]],
                 object_id_map: Dict[str, str], model: str, max_tokens: int,
                 temperature: float = 0.7, max_tries: int = 3,
                 output_folder: str = None, story_name: str = None) -> str:
    """
    Generate image-by-image analysis for a sequence of images.
    
    Args:
        client: OpenAI client instance
        images: List of PIL Image objects
        all_detections: List of detection lists for each image (includes both objects and landmarks)
        object_id_map: Dictionary mapping original object IDs to consistent IDs
        model: Name of the AI model to use
        max_tokens: Maximum number of tokens for the generated analysis
        temperature: Temperature for text generation
        max_tries: Maximum number of reprompt attempts if parsing fails
        output_folder: Path to folder where debug files will be saved
        story_name: Name of the movie/story for debug file naming
        
    Returns:
        tuple: (analysis_text, structured_analysis)
    """
    # Get valid setting elements for the prompt
    setting_elements = ", ".join([elem.value for elem in SettingElement])

    # Create debug folder if needed
    debug_folder = None
    if output_folder:
        debug_folder = Path(output_folder) / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)

    # Prefix for debug files
    prefix = f"{story_name}_" if story_name else ""

    # Prepare the system prompt with setting element options and object ID information
    system_prompt = f"""
    You are an expert film analyst. You will analyze a sequence of movie images, creating structured tables 
    for characters, objects, and settings. I've provided object detections and landmark recognitions for each image.
    
    Important information about the detected objects:
    1. Each detected object has a unique object ID (like "0-person-0", "1-car-5")
    2. I will provide a mapping from these object IDs to consistent entity IDs, which come in four types:
       - Characters: char1, char2, etc. (for people or characters)
       - Objects: obj1, obj2, etc. (for things)
       - Landmarks: lm1, lm2, etc. (for recognizable places/landmarks)
       - Background: bg1, bg2, etc. (for background elements)
    3. Objects with the same entity ID represent the same element appearing across multiple images
    4. Bounding box coordinates are provided in pixel values (actual image dimensions) as x1,y1,x2,y2
    5. Landmarks provide additional context about the setting and location
       - Landmarks are formatted as "landmark-ID: landmark-name: x1,y1,x2,y2"
       - If landmarks appear in multiple images, they will have entity IDs in the mapping section
    6. Some objects may be incorrectly classified or detected:
       - You can override object classifications if they're clearly wrong (e.g., a "dog" that is actually a "cat")
       - You can ignore detections if no actual object exists at that location
       - Use your visual understanding to correct any detection errors
       - Only include objects in your tables that you can actually see in the images
       - If you are sure there is an object in the image but it is not detected, you can add it to the table, remember to include the bounding box
       - If you detect two entities that are the same but have different IDs, you must merge them into one entity and use only one of the IDs

    
    For each image, provide:
    
    ## Image X
    
    ### Characters
    | Character ID | Name | Description | Emotions | Actions | Narrative Function | Bounding Box |
    |-------------|------|-------------|----------|---------|-------------------|--------------|
    
    IMPORTANT: If no characters are detected, you can omit the table and the section title. Bounding box is required.
    
    ### Objects
    | Object ID | Description | Function | Interaction | Narrative Function | Bounding Box |
    |-----------|-------------|----------|------------|-------------------|--------------|
    
    IMPORTANT: The landmarks and background elements should be included in the objects table.
    If no objects, landmarks, or background elements are detected, you can omit the table and the section title.
    Bounding box is required.
    
    ### Setting
    | Setting Element | Description | Mood | Time | Narrative Function |
    |-----------------|-------------|------|------|-------------------|
    
    IMPORTANT: For the Setting Element column, you must use only one of these specific categories:
    {setting_elements}
    
    After analyzing all images, provide a Narrative Structure table connecting the images:
    
    ## Narrative Structure
    | Narrative Phase | Description | Key Events | Images |
    |-----------------|-------------|-----------|--------|
    
    IMPORTANT: For the Narrative Structure table, the Narrative Phase column must use only one of these specific categories:
    Introduction, Development, Conflict, Turning Point, Conclusion
    The header ## Image X is required for every image even if there is no character or object table.
    
    Maintain consistent table formatting with proper markdown syntax.
    Each table must have headers and at least one row of data.
    Use the entity IDs (char1, obj2, etc.) I've provided rather than creating your own.
    """

    messages = [{
        "role": "system",
        "content": system_prompt
    }]

    # Process each image
    for i, image in enumerate(images):
        # Get detections for this image and filter out background objects that aren't matched across images
        detections = all_detections[i]
        filtered_detections = []

        for det in detections:
            # Include detection if:
            # 1. It's a landmark
            # 2. It's a foreground object (is_thing is True or None)
            # 3. It's a background object (is_thing is False) but is matched across images
            #    and belongs to a match with multiple detections
            if det.is_landmark or det.is_thing is None or det.is_thing or \
                    (det.get_global_id() in object_id_map and len(
                        [k for k, v in object_id_map.items() if v == object_id_map[det.get_global_id()]]) > 1):
                filtered_detections.append(det)

        # Create request message with image and filtered detections
        request_message = RequestMessage(
            image=image,
            detections=filtered_detections,
            image_idx=str(i + 1)
        )

        # Add to messages
        messages.append(request_message.to_openai_message())

        # Add information about object ID mapping (only for filtered detections)
        object_info = "Object ID mapping for this image (object ID → entity ID):\n"
        for det in filtered_detections:
            orig_id = det.get_global_id()
            if orig_id in object_id_map:
                consistent_id = object_id_map[orig_id]
                object_info += f"- {orig_id} → {consistent_id}\n"

        messages.append({
            "role": "user",
            "content": f"This is image {i + 1} of {len(images)}.\n{object_info}\nPlease update your analysis with this image."
        })

    # Request final analysis across all images
    messages.append({
        "role": "user",
        "content": f"""
        Now that you've seen all {len(images)} images, provide your complete image-by-image analysis with all tables as described.
        
        Remember:
        1. Use the consistent Character IDs (char1, cha2), Object IDs (obj1, obj1), Background IDs (bg1, g2) and Landmark IDs (lm1, lm2) that I provided
        2. Merge any duplicate entities with different IDs into one entity and use only one ID
        3. For the Setting table, the Setting Element column must use only one of these values per row:
           {setting_elements}
        4. You can invent character names using any proper names that fits the character in the context you should prefer proper names instead of "The man" or "The woman"
        5. The Bounding Box column should include the provided bounding box for the object in the image: x1,y1,x2,y2
        6. Include the final Narrative Structure table to connect all images
        """
    })

    analysis_text = None

    # Generate the analysis with reprompting if needed
    for attempt in range(1, max_tries + 1):
        # Save the request messages for debugging
        if debug_folder:
            request_file = debug_folder / f"{prefix}request_attempt_{attempt + 1}.txt"
            with open(request_file, 'w') as f:
                for msg in messages:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")

                    if isinstance(content, list):
                        # Handle multimodal content
                        content_parts = []
                        for item in content:
                            if item.get("type") == "image_url" or item.get("type") == "image":
                                content_parts.append("[IMG]\n")
                            elif item.get("type") == "text":
                                content_parts.append(item.get("text", "") + "\n")
                            else:
                                content_parts.append(str(item))
                        content_str = "\n".join(content_parts)
                    else:
                        content_str = content

                    f.write(f"--- {role.upper()} ---\n{content_str}\n\n")

        # Generate the analysis
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        analysis_text = response.choices[0].message.content

        # Try to parse the analysis
        error_message = ""
        try:
            success, error_message = validate_cot(analysis_text, images)

            # Check if we have successfully parsed all images
            if success:
                # Save successful attempt for comparison
                if debug_folder:
                    success_file = debug_folder / f"{prefix}successful_analysis.txt"
                    with open(success_file, 'w') as f:
                        f.write(analysis_text)
                return analysis_text
            else:
                print(f"Failed CoT analysis for story {story_name} (attempt {attempt + 1}): {error_message}")
                # Save failed attempt for comparison
                if debug_folder:
                    failed_file = debug_folder / f"{prefix}failed_analysis_attempt_{attempt + 1}.txt"
                    print(f"Debug file of attempt {attempt + 1} of story {story_name} saved to {failed_file}")
                    with open(failed_file, 'w') as f:
                        f.write(analysis_text)
        except Exception as e:
            print(f"Error parsing CoT analysis (attempt {attempt + 1}): {e}")
            # Save error details for debugging
            if debug_folder:
                error_file = debug_folder / f"{prefix}error_details_attempt_{attempt + 1}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error: {str(e)}\n\nAnalysis text:\n{analysis_text}")
                print(f"Saved error details to {error_file}")

        # If we get here, validation failed, configure the reprompt
        messages.append({
            "role": "assistant",
            "content": analysis_text
        })

        messages.append({
            "role": "user",
            "content": f"""
            Your analysis has formatting issues that make it difficult to parse. Please regenerate your analysis with these requirements:
            
            1. For each image, use the exact heading format "## Image X" (where X is the image number)
            2. Include all three tables (Characters, Objects, Setting) for each image, unless there are no objects or characters
            3. Use consistent table formatting with proper markdown headers and separators
            4. Ensure each table has at least one row of data
            5. In the Setting table, use ONLY these values for Setting Element column:
               {setting_elements}
            6. End with a Narrative Structure table
            7. Do not omit any images
            8. The bounding boxes are always required and in the format x1,y1,x2,y2. If you don't see an object don't include it in the table
            
            The error found was {error_message}
            Please regenerate the complete analysis following these formatting requirements.
            """
        })

        print(f"Trying reprompting for CoT analysis (attempt {attempt + 1})")

    # If we've exhausted reprompt attempts, return the last analysis
    return analysis_text


def generate_story(client, analysis_text: str, structured_analysis: CoT,
                   images: List[Image.Image], model: str, max_tokens: int,
                   temperature: float = 0.7, max_tries: int = 10) -> str:
    """
    Generate a grounded story based on the image analysis with appropriate tags.
    
    Args:
        client: OpenAI client instance
        analysis_text: The image-by-image analysis text
        structured_analysis: The structured analysis data
        images: The images used for the analysis
        model: Name of the AI model to use
        max_tokens: Maximum number of tokens for the generated story
        temperature: Temperature for text generation
        max_tries: Maximum number of attempts for the generated story
        
    Returns:
        str: The generated grounded story with tags
    """
    # Create a simplified character mapping for the prompt
    character_map = {}
    object_map = {}
    landmark_map = {}
    background_map = {}

    for image in structured_analysis.images:
        for char in image.characters:
            if char.character_id not in character_map and char.name:
                character_map[char.character_id] = char.name

        for obj in image.objects:
            if obj.object_id not in object_map and not (hasattr(obj, 'is_landmark') and obj.is_landmark):
                object_map[obj.object_id] = obj.description

            # If the object is a landmark
            if hasattr(obj, 'is_landmark') and obj.is_landmark and obj.object_id not in landmark_map:
                landmark_map[obj.object_id] = obj.description

            # If the object is a background element
            if hasattr(obj, 'is_thing') and not obj.is_thing and obj.object_id not in background_map:
                background_map[obj.object_id] = obj.description

        for setting in image.settings:
            setting_key = setting.setting_element.value.upper().replace(" ", "_")
            if setting_key not in background_map:
                background_map[setting_key] = setting.description

    # Prepare character, object, and setting information for the prompt
    character_info = "\n".join([f"{cid}: {name}" for cid, name in character_map.items()])
    object_info = "\n".join([f"{oid}: {desc}" for oid, desc in object_map.items()])
    landmark_info = "\n".join([f"{lid}: {desc}" for lid, desc in landmark_map.items()])
    background_info = "\n".join([f"{bid}: {desc}" for bid, desc in background_map.items()])

    system_prompt = """
    You are a creative storyteller who creates vivid, creative, and inspiring stories based on visual scenes.
    Using the image-by-image analysis provided, craft a compelling, creative story that's plausible given 
    the visual elements, characters, and settings shown in the images.
    
    Important: 
    - Your story should NOT be a simple description of what's in each image
    - Create an engaging narrative with plot, character development, and emotional depth
    - Feel free to invent character names, backstories, motivations, and relationships
    - You can create any type of story (drama, mystery, romance, adventure, etc.) that's plausible given the visual elements but it should engage the reader
    - Your narrative can be completely different from any original movie these images might be from
    - Avoid repetitions of nouns, preferring to use <gdo char1 char2 char3>They</gdo> instead of <gdo char1>John</gdo> and <gdo char2>Mary</gdo> and <gdo char3>Bob</gdo> in the same sentence
    - You don't have to use every object in the tables in your story, but you should use the most relevant ones
    
    Your story should use special grounding tags to reference elements from the analysis:

    
    1. Image grounding: <gdf image1>Text describing events in image 1</gdf>
        Each part of the story must be inside a image tag indicating which image it describes
    
    2. Character and action tags:
        For character references: <gdo char1>Character name, pronoun or description</gdo> or <gdo char1 char2>They</gdo> for multiple characters
        For character actions: <gda char1>action description</gda> or <gda char1 char2>action description</gda> for multiple characters

    3. Object grounding: <gdo obj1>Object reference</gdo> or <gdo obj1 obj2>Objects reference</gdo>
        Use this for specific objects in the scene
    
    4. Landmark grounding: <gdl lm1>Landmark description</gdl> or <gdl lm1 lm2>Landmarks description</gdl>
        Use this for landmarks or recognizable locations
   
    5. Background grounding: <gdl bg1>Background element description</gdl> or <gdl bg1 bg2>Background elements description</gdl>
        Use this for background elements or general settings
    
    Example of properly formatted text:
    <gdi image1>
    <gdo char1>Sarah</gdo> <gda char1>held</gda> 
    <gdo obj3>the ancient book</gdo> as <gdo char1>she</gdo> <gda char1>gazed</gda> across 
    <gdl lm1>the famous cliffs</gdl> rising above
    <gdl bg1>the misty shoreline</gdl>.
    </gdi>
    
    <gdi image2>
    The wind picked up as <gdo char1 char2>they</gdo> <gda char1 char2>walked</gda> toward 
    <gdl lm2>the abandoned lighthouse</gdl> <gda char1 char2>standing</gda> on
    <gdl bg2 bg3>the rocky shoreline</gdl>.
    </gdi>
    
    Create a rich, inspiring or suspense story that's plausible given the visual elements in the images,
    making sure every part of the text is within the appropriate image tag.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"""
            Here is the image-by-image analysis:
            
            {analysis_text}
            
            Use the following elements in your story with their corresponding tags:
            
            CHARACTERS:
            {character_info}
            
            OBJECTS:
            {object_info}
            
            LANDMARKS:
            {landmark_info}
            
            BACKGROUND ELEMENTS:
            {background_info}
            
            I'll also include the actual images for reference. Create a creative and engaging grounded story based on this analysis. Remember:
            1. Create a rich story with interesting character development, that contais suspense or is inspiring
            2. Feel free to invent character names, backstories, and dialogues that suit their appearance and actions
            3. You can speculate on what characters are thinking or saying - create rich dialogue between characters
            4. Include characters' internal thoughts, emotions, and motivations to add depth
            5. Your story can be completely different from any original movie - just make it plausible based on the visual elements
            6. Wrap each part of the story in <gdi image#> tags
            7. Use the appropriate tags for different elements:
               - <gdo char#> for character references (the person name or pronoun)
               - <gda char#> for actions (what they're doing)
               - <gdo obj#> for objects
               - <gdl lm#> for landmarks
               - <gdl bg#> for background elements
               where char#, obj#, lm#, bg# are the IDs from the analysis tables
            8. Follow the format shown in the examples, for multiple entities in a single reference add all the ids as attributes in the tag: <gdo char1 char2>They</gdo>
            9. Every part of your story must be inside an image tag
            10. Ground every reference of a character, object, or setting including pronouns. Every pronoun such as he his she her should be grounded in a <gdo char#> tag.
            11. Use the analysis tables to guide your story, but don't just describe the images.
            """
        }
    ]

    # Add images as separate user messages
    for i, image in enumerate(images):
        # Create a byte buffer for the image
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                },
                {
                    "type": "text",
                    "text": f"Image {i + 1}"
                }
            ]
        })

    # Add final message asking for the story
    messages.append({
        "role": "user",
        "content": "Now, based on all the images and the analysis, create the creative grounded story as described."
    })


    story_text = None

    # Generate the story with reprompting if needed
    for attempt in range(1, max_tries + 1):
        # Generate the story
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

        story_text = response.choices[0].message.content
    
    
        # Validate the story
        error_message = ""
        try:
            success, error_message = validate_story(analysis_text, story_text, images)
    
            # Check if we have successfully generated a valid story
            if success:
                return story_text
            
        except Exception as e:
            print(f"Error validating story (attempt {attempt + 1}): {e}")
            
        print(f"Trying reprompting for story (attempt {attempt + 1})")

    return story_text


def validate_cot(cot_analysis_text: str, images: List[Image]) -> Tuple[bool, str]:
    """
    Validate the chain of thought according to the rules.
    
    Args:
        cot_analysis_text: The image-by-image analysis text (CoT)
        images: The list of images used for the analysis
        
    Returns:
        tuple: (is_valid, error_message)
    """

 
    # Validates the structure of the bounding box string
    def validate_bounding_box(bbox_str: str, image: Image):
        if not bbox_str or bbox_str.lower() == 'none' or bbox_str.strip() == '':
            return False

        # Check format: four numbers separated by commas
        parts = bbox_str.split(',')
        if len(parts) != 4:
            return False

        # Verify all parts are numeric
        try:
            for part in parts:
                int(part.strip())
        except ValueError:
            return False

        # Check if the bounding box is within the image dimensions
        try:
            x1, y1, x2, y2 = map(int, parts)
            width, height = image.size
            if x1 < 0 or y1 < 0 or x2 > width * 1.1 or y2 > height * 1.1:
                return False
        except ValueError:
            return False
        
        # Check if the bounding box is valid (x1 < x2 and y1 < y2)
        if x1 >= x2 or y1 >= y2:
            return False

        return True


    try:
        num_images = len(images)

        # 1. Parse the analysis text
        complete_analysis = StoryReasoningUtil.parse_cot(cot_analysis_text)
        cot_headers_count = len(complete_analysis.images)
        
        
        # 2. Check if the number of images matches the number of image sections in the CoT
        if num_images != cot_headers_count:
            return False, f"Count mismatch: Images={num_images}, Parsed sections={cot_headers_count}"

        # 3. Parse each image section
        for image_num in range(1, num_images + 1):
            image_analysis = StoryReasoningUtil.parse_image(cot_analysis_text, image_num)

            if image_analysis is None:
                return False, f"Missing analysis for image {image_num}"

            # Check if the tables, in case they exist have valid data
            if image_analysis.characters:
                for char in image_analysis.characters:
                    if char.character_id is None or not char.character_id.startswith("char"):
                        return False, f"Image {image_num}: Incorrect Character ID for character {char.character_id}"

            if image_analysis.objects:
                for obj in image_analysis.objects:
                    if obj.object_id is None or not (obj.object_id.startswith("obj") or obj.object_id.startswith("lm") or obj.object_id.startswith("bg")):
                        return False, f"Image {image_num}: Incorrect Object ID for object {obj.object_id}"

            if not image_analysis.settings:
                return False, f"Image {image_num}: Missing Setting table"

            # Check if all bounding boxes are valid
            for char in image_analysis.characters:
                if not validate_bounding_box(char.bounding_box, images[image_num - 1]):
                    return False, f"Image {image_num}: Invalid bounding box format or value for character {char.character_id} {char.bounding_box}"

            # Check bounding boxes for objects
            for obj in image_analysis.objects:
                if not validate_bounding_box(obj.bounding_box, images[image_num - 1]):
                    return False, f"Image {image_num}: Invalid bounding box format or value for object {obj.object_id} {obj.bounding_box}"


        # 4. Parse narrative structure
        narrative_structure = StoryReasoningUtil.parse_narrative_structure(cot_analysis_text)

        if narrative_structure is None or not narrative_structure.phases:
            return False, "Failed to parse Narrative Structure section"

        # Check if all required narrative phases are present
        found_phases = []
        for phase in narrative_structure.phases:
            phase_value = phase.get('Narrative Phase')
            if isinstance(phase_value, NarrativePhase):
                found_phases.append(phase_value.value)
            else:
                found_phases.append(str(phase_value))

        missing_phases = []
        required_phases = ['Introduction', 'Development', 'Conflict', 'Turning Point', 'Conclusion']
        for phase in required_phases:
            if phase not in found_phases and phase not in [p.value for p in found_phases]:
                missing_phases.append(phase)

        if missing_phases:
            return False, f"Missing narrative phases: {', '.join(missing_phases)}"


        # All validation passed
        return True, "CoT validation successful"

    except Exception as e:
        return False, f"CoT validation error: {str(e)}"


def validate_story(cot_text: str, story_text : str, images: List[Image]):
    """
    Validate the generated story according to the rules.
    
    Args:
        cot_text: The image-by-image analysis text (CoT)
        story_text: The generated story with tags
        images: The list of images used for the analysis
        
    Returns:
        tuple: (is_valid, error_message)
    """

    # Count the number of '<gdi image*>' tags in the story
    def count_gdi_tags(text):
        pattern = r'<gdi image\d+>'
        matches = re.findall(pattern, text)
        return len(matches)
        

    # Extract all grounding tag IDs from story
    def extract_grounding_ids(text):
        char_pattern = r'<gd[oa] (.*?)>'
        all_ids = []

        matches = re.findall(char_pattern, text)
        for match in matches:
            # Split IDs if multiple are present (e.g., "char1 char2")
            ids = match.split()
            all_ids.extend(ids)

        return set(all_ids)
    

    try:
        num_images = len(images)
        
        # 1. Parse the analysis text
        complete_analysis = StoryReasoningUtil.parse_cot(cot_text)

        # 2. Check if the number of <gdi> tags in the story is equal to the number of images
        gdi_tags_count = count_gdi_tags(story_text)

        if num_images != gdi_tags_count:
            return False, f"Count mismatch: Images={num_images}, <gdi> tags={gdi_tags_count}"

        # 3. Validate that all character IDs in story tags also appear in the CoT
        cot_ids = set()
        for image in complete_analysis.images:
            for char in image.characters:
                cot_ids.add(char.character_id)
            for obj in image.objects:
                cot_ids.add(obj.object_id)

        # Get all IDs used in story's grounding tags
        story_tag_ids = extract_grounding_ids(story_text)

        # Find IDs in story that aren't in CoT
        missing_ids = [id for id in story_tag_ids if id not in cot_ids]

        if missing_ids:
            return False, f"Story uses IDs not found in analysis: {', '.join(missing_ids)}"

        # All validation passed
        return True, "Validation successful"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


def process_story(client, story, output_folder: str, model: str,
                  max_tokens: int, temperature: float = 0.7, max_tries: int = 3,
                  landmark_llm_base_url: str = None, landmark_llm_api_key: str = None, landmark_model: str = None,
                  google_credentials_path: str = None, story_idx: int = None):
    """
    Process a story from the StoryReasoningDerivedDataset to generate image analysis and a grounded story.
    
    Args:
        client: OpenAI client instance used for CoT and story generation
        story: Dictionary containing story data from StoryReasoningDerivedDataset
        output_folder: Path to folder where results will be saved
        model: Name of the AI model to use
        max_tokens: Maximum tokens for generation
        temperature: Temperature for text generation
        max_tries: Maximum number of reprompt attempts if parsing fails
        landmark_llm_base_url: Base URL for LLM landmark detector
        landmark_llm_api_key: API key for the LLM landmark detector
        landmark_model: Name of model to use for landmark detection
        google_credentials_path: Path to Google Cloud credentials for landmark detection
        story_idx: Index of the story in the dataset
    """
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if files for this index already exist
    cot_file = output_path / f"{story_idx}_cot.txt"
    story_file = output_path / f"{story_idx}_story.txt"
    metadata_file = output_path / f"{story_idx}_metadata.txt"

    # Skip if all files exist
    if cot_file.exists() and story_file.exists() and metadata_file.exists():
        print(f"Skipping story {story_idx} - files already exist")
        return True

    images = story['images']
    image_ids = story['image_ids']

    # Extract story name from first image ID for debug file naming
    story_name = Path(image_ids[0]).stem.split('_')[0] if image_ids else "unknown"

    # Detect and match objects in images
    print(f"Story {story_idx}: Detecting and matching objects...")
    all_detections, object_id_map = detect_and_match_objects(images)

    # Detect landmarks if credentials or API key provided
    if google_credentials_path or landmark_llm_api_key:
        print(f"Story {story_idx}: Detecting landmarks...")
        all_landmark_detections = detect_landmarks(images, google_credentials_path, landmark_llm_base_url, landmark_model, landmark_llm_api_key)

        # Merge object and landmark detections
        all_detections = merge_detections(all_detections, all_landmark_detections)
        
        
    # Generate the CoT
    print(f"Story {story_idx}: Generating CoT analysis...")
    analysis_text = generate_cot(
        client, images, all_detections, object_id_map, model,
        max_tokens, temperature, max_tries, output_folder, story_name
    )
    structured_analysis = StoryReasoningUtil.parse_cot(analysis_text)
    
    if not analysis_text:
        print(f"Story {story_idx}: CoT generation failed. Skipping story.")
        return False


    print(f"Story {story_idx}: Generating grounded story...")
    story_text = generate_story(
        client, analysis_text, structured_analysis, images, model, max_tokens, temperature
    )
    
    if not story_text:
        print(f"Story {story_idx}: Story generation failed. Skipping story.")
        return False

  

    print(f"Story {story_idx}: Completed successfully!")

    # Save analysis text (Chain of Thought)
    with open(cot_file, 'w') as f:
        f.write(analysis_text)

    # Save story
    with open(story_file, 'w') as f:
        f.write(story_text)

    # Save metadata (image IDs)
    with open(metadata_file, 'w') as f:
        f.write("\n".join(image_ids))

    print(f"Story {story_idx}: Files saved successfully")
    return True


# Updated worker function that creates its own dataset instance
def process_story_worker(idx, gpu_ids, hf_repo, min_images_per_story, 
                         server_domain, api_key, model, 
                         landmark_server_domain, landmark_api_key, landmark_model,
                         output_path, max_tokens, temperature, max_tries,
                         google_credentials_path):
    """Worker function that loads its own dataset and processes a single story."""
    try:
        # Assign a GPU to this process based on process ID
        proc_id = multiprocessing.current_process()._identity[0] - 1  # Process IDs start at 1
        if gpu_ids:
            # Assign GPU in round-robin fashion if multiple GPUs are available
            gpu_id = gpu_ids[proc_id % len(gpu_ids)]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
        # Check if files for this index already exist
        output_path_obj = Path(output_path)
        cot_file = output_path_obj / f"{idx}_cot.txt"
        story_file = output_path_obj / f"{idx}_story.txt"
        metadata_file = output_path_obj / f"{idx}_metadata.txt"

        # Skip if all files exist
        if cot_file.exists() and story_file.exists() and metadata_file.exists():
            # print(f"Skipping story {idx} - files already exist")
            return True

        print(f"Process {proc_id} generating story {idx}")
        
        # Each worker loads its own dataset copy
        # This is more memory-efficient in multiprocessing as each process has its own memory space
        from story_reasoning.datasets.story_reasoning.story_reasoning_derived_dataset import StoryReasoningDerivedDataset

        worker_dataset = StoryReasoningDerivedDataset(
            hf_repo=hf_repo,
            transform=None,
            min_images_per_story=min_images_per_story,
        )

        # Each worker creates its own client
        generation_client = OpenAI(
            base_url=f"{server_domain}/v1" if server_domain is not None else None,
            api_key=api_key,
            timeout=Timeout(1800)
        )

        # Get story from dataset
        story = worker_dataset[idx]

        return process_story(
            generation_client,
            story,
            output_path,
            model,
            max_tokens,
            temperature,
            max_tries,
            landmark_server_domain,
            landmark_api_key,
            landmark_model,
            google_credentials_path,
            idx
        )
    except Exception as e:
        print(f"Error processing story {idx}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate image analysis and grounded story from StoryReasoningDerivedDataset")
    parser.add_argument("output_path", help="Path to folder where results will be saved")
    parser.add_argument("--server_domain", help="Domain of the vLLM server", default=None)
    parser.add_argument("--api_key", help="API key for OpenAI", default="not-needed")
    parser.add_argument("--hf_repo", help="Hugging Face repo for dataset", default="daniel3303/GroundCap")
    parser.add_argument("--min_images", type=int, default=5, help="Minimum images per story")
    parser.add_argument("--story_idx", type=int, default=None, help="Index of specific story to process (optional)")
    parser.add_argument("--model", default="gpt-4o", help="Name of the vision-language model to use")
    parser.add_argument("--max_tokens", type=int, default=18000, help="Maximum tokens for generated content")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for text generation")
    parser.add_argument("--max_tries", type=int, default=10, help="Maximum reprompt attempts if parsing the narrative structure fails")
    parser.add_argument("--google_credentials_path", help="Path to Google Cloud credentials for landmark detection", default=None)
    parser.add_argument("--max_stories", type=int, default=None, help="Maximum number of stories to process")
    parser.add_argument("--num_processes", type=int, default=2, help="Number of parallel processes to use")
    parser.add_argument("--gpus", help="Comma-separated list of GPU IDs to use", default=None)
    parser.add_argument("--landmark_server_domain", help="Domain of the vLLM server to perform landmark recognition. If None the --server_domain will be used.", default=None)
    parser.add_argument("--landmark_api_key", help="API key for OpenAI for landmark recognition. If None the --api_key will be used.", default=None)
    parser.add_argument("--landmark_model", help="Name of the vision-language model to use for the landmark recognition. If none the --model will be used.", default=None)

    args = parser.parse_args()

    gpu_ids = None
    if args.gpus:
        gpu_ids = [id.strip() for id in args.gpus.split(',')]
        print(f"Using GPUs: {gpu_ids}")


    # Credentials used to connect to the landmark recognition LLM. By default, the same as the ones used to connect to the story generation LLM
    landmark_server_domain = args.landmark_server_domain if args.landmark_server_domain is not None else args.server_domain
    landmark_api_key = args.landmark_api_key if args.landmark_api_key is not None else args.api_key
    landmark_model = args.landmark_model if args.landmark_model is not None else args.model
    

    # Process a specific story if story_idx is provided
    if args.story_idx is not None:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
            print(f"Using GPU {gpu_ids[0]} for single story processing")
            
        # Load dataset for single story processing
        print(f"Loading StoryReasoningDerivedDataset from {args.hf_repo}...")
        story_dataset = StoryReasoningDerivedDataset(
            hf_repo=args.hf_repo,
            transform=None,
            min_images_per_story=args.min_images,
        )
        
        if args.story_idx < 0 or args.story_idx >= len(story_dataset):
            print(
                f"Error: Story index {args.story_idx} is out of range. Dataset contains {len(story_dataset)} stories.")
            return

        story = story_dataset[args.story_idx]
        print(f"Processing single story {args.story_idx} with {len(story['images'])} images")

        # Create client for single story processing
        client = OpenAI(
            base_url=f"{args.server_domain}/v1" if args.server_domain is not None else None,
            api_key=args.api_key,
            timeout=Timeout(1800)
        )
    
        process_story(
            client,
            story,
            args.output_path,
            args.model,
            args.max_tokens,
            args.temperature,
            args.max_tries,
            landmark_server_domain,
            landmark_api_key,
            landmark_model,
            args.google_credentials_path,
            args.story_idx
        )
    else:
        print(f"Loading StoryReasoningDerivedDataset from {args.hf_repo} to get story count...")
        temp_dataset = StoryReasoningDerivedDataset(
            hf_repo=args.hf_repo,
            transform=None,
            min_images_per_story=args.min_images,
        )
        dataset_length = len(temp_dataset)

        # Generate indices for the stories to process
        indices = list(range(dataset_length))
        random.shuffle(indices)

        # Limit number of stories if specified
        if args.max_stories is not None:
            indices = indices[:args.max_stories]

        print(f"Processing {len(indices)} stories with {args.num_processes} parallel processes")

        # Create output folder if it doesn't exist
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a partial function with fixed arguments
        worker_fn = partial(
            process_story_worker,
            hf_repo=args.hf_repo,
            gpu_ids=gpu_ids,
            min_images_per_story=args.min_images,
            server_domain=args.server_domain,
            api_key=args.api_key,
            model=args.model,
            landmark_server_domain = landmark_server_domain,
            landmark_api_key = landmark_api_key,
            landmark_model = landmark_model,
            output_path=args.output_path,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_tries=args.max_tries,
            google_credentials_path=args.google_credentials_path
        )

        # Use multiprocessing to process stories in parallel
        num_processes = min(args.num_processes, len(indices))
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(worker_fn, indices)

        # Count successes and failures
        successes = sum(1 for r in results if r)
        failures = len(results) - successes

        print("\nProcessing complete!")
        print(f"Successfully processed: {successes} stories")
        print(f"Failed to process: {failures} stories")


if __name__ == "__main__":
    main()
