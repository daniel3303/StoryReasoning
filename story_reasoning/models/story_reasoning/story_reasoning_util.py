import re
from typing import Union, Tuple, List, Dict

from story_reasoning.models.story_reasoning.character import Character
from story_reasoning.models.story_reasoning.image_analysis import ImageAnalysis
from story_reasoning.models.story_reasoning.narrative_phase import NarrativePhase
from story_reasoning.models.story_reasoning.narrative_structure import NarrativeStructure
from story_reasoning.models.story_reasoning.object import Object
from story_reasoning.models.story_reasoning.setting import SettingElement, Setting
from story_reasoning.models.story_reasoning.cot import CoT
from story_reasoning.models.story_reasoning.story import Story
from story_reasoning.models.story_reasoning.story_image import StoryImage
import copy


class StoryReasoningUtil:
    """
    Parser for extracting structured data from image analysis.
    """

    @staticmethod
    def _parse_table(table_text) -> List[Dict[str, str]]:
        """Parse a markdown table into rows and headers"""
        lines = table_text.strip().split('\n')

        # Extract headers
        header_line = lines[0]
        headers = [h.strip() for h in re.findall(r'\|(.*?)(?=\|)', header_line + '|')]
        headers = [h for h in headers if h and not all(c == '-' for c in h)]  # Remove empty or separator headers

        # Skip the separator line
        data_lines = lines[2:] if len(lines) > 2 else []

        rows = []
        for line in data_lines:
            if not line.strip() or '|' not in line:
                continue

            # Extract cell values
            cells = [cell.strip() for cell in re.findall(r'\|(.*?)(?=\|)', line + '|')]
            if len(cells) >= len(headers):
                row_data = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row_data)

        return rows

    @staticmethod
    def _extract_image_sections(text, image_num) -> Tuple[Union[str, None], Union[str, None], Union[str, None], bool]:
        """Extract the Characters, Objects, and Setting sections for a specific frame"""
        image_pattern = rf"## Image {image_num}\s*\n"
        next_frame_pattern = r"## Image \d+\s*\n"

        # Find the current frame section
        image_match = re.search(image_pattern, text)
        if not image_match:
            return None, None, None, False

        curr_image_start_end = image_match.end()

        # Find the next frame section
        next_image_match = re.search(next_frame_pattern, text[curr_image_start_end:])
        curr_image_end = curr_image_start_end + next_image_match.start() if next_image_match else len(text)

        image_content = text[curr_image_start_end:curr_image_end]

        # Extract each section
        character_pattern = r"### Characters\s*\n((?:.+\n)+)"
        object_pattern = r"### Objects\s*\n((?:.+\n)+)"
        setting_pattern = r"### Setting\s*\n((?:.+\n)+)"

        character_match = re.search(character_pattern, image_content)
        object_match = re.search(object_pattern, image_content)
        setting_match = re.search(setting_pattern, image_content)
        

        character_table = character_match.group(1) if character_match else None
        object_table = object_match.group(1) if object_match else None
        setting_table = setting_match.group(1) if setting_match else None

        return character_table, object_table, setting_table, True

    @staticmethod
    def _extract_narrative_structure(text) -> Union[str, None]:
        """Extract the Narrative Structure section"""
        narrative_pattern = r"## Narrative Structure\s*\n((?:.+(?:\n|$))+)"
        narrative_match = re.search(narrative_pattern, text)

        if not narrative_match:
            return None

        return narrative_match.group(1)

    @staticmethod
    def parse_image(text, image_num) -> Union[ImageAnalysis, None]:
        """Parse a single frame's analysis into structured data"""
        character_table, object_table, setting_table, found_content = StoryReasoningUtil._extract_image_sections(text, image_num)
        
        if not found_content:
            return None

    
        # Parse characters
        characters = []
        if character_table:
            character_rows = StoryReasoningUtil._parse_table(character_table)
            for row in character_rows:
                try:
                    character = Character(
                        character_id=row.get('Character ID', ''),
                        name=row.get('Name', ''),
                        description=row.get('Description', ''),
                        emotions=row.get('Emotions', ''),
                        actions=row.get('Actions', ''),
                        narrative_function=row.get('Narrative Function', ''),
                        bounding_box=row.get('Bounding Box', '')
                    )
                    characters.append(character)
                except Exception as e:
                    print(f"Error parsing character in image {image_num}: {e}")

        # Parse objects
        objects = []
        if object_table:
            object_rows = StoryReasoningUtil._parse_table(object_table)
            for row in object_rows:
                try:
                    obj = Object(
                        object_id=row.get('Object ID', ''),
                        description=row.get('Description', ''),
                        function=row.get('Function', ''),
                        interaction=row.get('Interaction', ''),
                        narrative_function=row.get('Narrative Function', ''),
                        bounding_box=row.get('Bounding Box', '')
                    )
                    objects.append(obj)
                except Exception as e:
                    print(f"Error parsing object in image {image_num}: {e}")

        # Parse settings
        settings = []
        if setting_table:
            setting_rows = StoryReasoningUtil._parse_table(setting_table)
            for row in setting_rows:
                try:
                    # Convert string setting_element to enum
                    setting_element_str = row.get('Setting Element', '')
                    try:
                        setting_element = SettingElement[setting_element_str.upper().replace(' ', '_')]
                    except (KeyError, ValueError):
                        # Find the closest match
                        valid_elements = list(SettingElement)
                        for elem in valid_elements:
                            if elem.value.lower() in setting_element_str.lower():
                                setting_element = elem
                                break
                        else:
                            # Default to Location if no match found
                            setting_element = SettingElement.LOCATION
    
                    setting = Setting(
                        setting_element=setting_element,
                        description=row.get('Description', ''),
                        mood=row.get('Mood', ''),
                        time=row.get('Time', ''),
                        narrative_function=row.get('Narrative Function', '')
                    )
                    settings.append(setting)
                except Exception as e:
                    print(f"Error parsing setting in image {image_num}: {e}")

        return ImageAnalysis(
            image_number=image_num,
            characters=characters,
            objects=objects,
            settings=settings
        )

    @staticmethod
    def parse_narrative_structure(text) -> Union[NarrativeStructure, None]:
        """Parse the narrative structure table into structured data"""
        narrative_table = StoryReasoningUtil._extract_narrative_structure(text)
        if not narrative_table:
            return None
    
        rows = StoryReasoningUtil._parse_table(narrative_table)
    
        # Convert narrative phase strings to enum values
        for row in rows:
            if 'Narrative Phase' in row:
                phase_str = row['Narrative Phase']
                try:
                    # Try direct matching
                    row['Narrative Phase'] = NarrativePhase[phase_str.upper().replace(' ', '_')]
                except (KeyError, ValueError):
                    # Find closest match
                    for phase in NarrativePhase:
                        if phase.value.lower() in phase_str.lower():
                            row['Narrative Phase'] = phase
                            break
                    else:
                        # Default to Introduction if no match found
                        row['Narrative Phase'] = NarrativePhase.INTRODUCTION
    
        return NarrativeStructure(phases=rows)

    @staticmethod
    def parse_cot(text) -> CoT:
        """Parse the complete analysis text into structured data"""
        complete_analysis = CoT()
        
        curr_image = 1
        while True:
            image_analysis = StoryReasoningUtil.parse_image(text, curr_image)
            if image_analysis:
                complete_analysis.images.append(image_analysis)
            curr_image += 1
            if not image_analysis:
                break
                        
        complete_analysis.narrative_structure = StoryReasoningUtil.parse_narrative_structure(text)

        return complete_analysis

    @staticmethod
    def strip_story_tags(story_text: str) -> str:
        """
        Remove grounding tags from story text and return the cleaned text.
        """
        
        # Remove image tags (<gdi image#>...</gdi>)
        text = re.sub(r'<gdi image\d+>', '', story_text)
        text = re.sub(r'</gdi>', '', text)
    
        # Remove other grounding tags (<gdo...>, <gda...>, <gdl...>)
        text = re.sub(r'<gd[oal][^>]*>', '', text)
        text = re.sub(r'</gd[oal]>', '', text)
    
        # Clean up any extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
    
        return text

    @staticmethod
    def extract_cot_text(text: str) -> str:
        """
        Extract chain of thought (CoT) text from within <think></think> tags in the model output.
        
        Args:
            text: The full model output containing <think> tags
            
        Returns:
            The CoT text inside the <think></think> tags, or empty string if not found
        """
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, text, re.DOTALL)
    
        if match:
            return match.group(1).strip()
        return ""
    
    @staticmethod
    def extract_story_text(text: str) -> str:
        """
        Extract the story text from the model output after the </think> tag.
        
        Args:
            text: The full model output containing <think> tags
            
        Returns:
            The story text after the </think> tag, or the original text if no tags found
        """
        # Find position of </think> tag
        end_think_pos = text.find('</think>')
    
        if end_think_pos != -1:
            # Return everything after the closing tag
            story_text = text[end_think_pos + 9:].strip()  # +9 to account for </think> length
            return story_text
    
        # If no </think> tag is found
        # - If a <think> tag is found then return an empty text becasue the model didn't start generating the story
        # - If no <think> tag is found then return the original text as the model didn't generate a chain of thought
        if '<think>' in text:
            return ""
        return text

    @staticmethod
    def cot_to_string(cot) -> str:
        """
        Exports the CoT object back to a markdown string in the original format.
        
        Args:
            cot: CoT object containing structured data
            
        Returns:
            String representation of the analysis in the original format
        """
        output = []
    
        # Export each image analysis
        for image in cot.images:
            output.append(f"## Image {image.image_number}")
            output.append("")
    
            # Characters section
            output.append("### Characters")
            output.append("| Character ID | Name | Description | Emotions | Actions | Narrative Function | Bounding Box |")
            output.append("|--------------|------|-------------|----------|---------|-------------------|--------------|")
            for char in image.characters:
                output.append(f"| {char.character_id} | {char.name} | {char.description} | {char.emotions} | {char.actions} | {char.narrative_function} | {char.bounding_box} |")
            output.append("")
    
            # Objects section
            output.append("### Objects")
            output.append("| Object ID | Description | Function | Interaction | Narrative Function | Bounding Box |")
            output.append("|-----------|-------------|----------|------------|-------------------|--------------|")
            for obj in image.objects:
                output.append(f"| {obj.object_id} | {obj.description} | {obj.function} | {obj.interaction} | {obj.narrative_function} | {obj.bounding_box} |")
            output.append("")
    
            # Settings section
            output.append("### Setting")
            output.append("| Setting Element | Description | Mood | Time | Narrative Function |")
            output.append("|-----------------|-------------|------|------|-------------------|")
            for setting in image.settings:
                output.append(f"| {setting.setting_element.value} | {setting.description} | {setting.mood} | {setting.time} | {setting.narrative_function} |")
    
            output.append("")
            output.append("")
    
        # Export narrative structure
        if cot.narrative_structure:
            output.append("## Narrative Structure")
            output.append("")
            output.append("| Narrative Phase | Description | Key Events | Images |")
            output.append("|-----------------|-------------|-----------|--------|")
            for phase in cot.narrative_structure.phases:
                narrative_phase = phase.get('Narrative Phase', '')
                phase_name = narrative_phase.value if hasattr(narrative_phase, 'value') else narrative_phase
                description = phase.get('Description', '')
                key_events = phase.get('Key Events', '')
                images = phase.get('Images', '')
                output.append(f"| {phase_name} | {description} | {key_events} | {images} |")
    
        return "\n".join(output)



    @staticmethod
    def parse_story(story_text: str) -> Story:
        """
        Parse a story with <gdi> tags into a structured Story object.
        
        Args:
            story_text: Text containing <gdi> tags with image content
            
        Returns:
            Story object containing structured images and text
        """
        # Find all image sections with their content
        image_pattern = r'<gdi image(\d+)>(.*?)</gdi>'
        matches = re.findall(image_pattern, story_text, re.DOTALL)

        story = Story()

        for image_num_str, image_content in matches:
            # Create a StoryImage for each match
            image_number = int(image_num_str)
            image_text = image_content.strip()
            story.images.append(StoryImage(image_number=image_number, text=image_text))

        # Sort images by number to ensure proper sequence
        story.images.sort(key=lambda x: x.image_number)

        return story

    @staticmethod
    def story_to_string(story: Story) -> str:
        """
        Export a Story object back to the original tagged format.
        
        Args:
            story: Story object containing structured images and text
            
        Returns:
            String representation with the original tags
        """
        output = []

        for image in story.images:
            output.append(f"<gdi image{image.image_number}>")
            output.append(image.text)
            output.append("</gdi>")
            output.append("")  # Add empty line between images

        return "\n".join(output).strip()


    @staticmethod
    def filter_unmentioned_entities(cot: CoT, story: Story, filter_characters: bool = True, filter_objects: bool = True, filter_backgrounds: bool = True) -> CoT:
        """
        Creates a new CoT containing only selected elements whose IDs are mentioned in the story,
        with options to control filtering for different element types.
        
        Args:
            cot: The original CoT object with all characters and objects
            story: Story object containing text with element references
            filter_characters: Whether to filter characters (char# IDs). If False, all characters are retained.
            filter_objects: Whether to filter objects (obj# IDs). If False, all objects are retained.
            filter_backgrounds: Whether to filter background elements (bg# IDs). If False, all backgrounds are retained.
            
        Returns:
            A new filtered CoT object
        """
    
        # Extract all character and object IDs mentioned in the story
        mentioned_ids = StoryReasoningUtil._extract_mentioned_ids(story)
    
        # Create a deep clone of the original CoT
        filtered_cot = copy.deepcopy(cot)
    
        # For each image analysis in the clone, filter elements based on parameters
        for image_analysis in filtered_cot.images:
            # Filter characters if requested
            if filter_characters:
                image_analysis.characters = [
                    char for char in image_analysis.characters
                    if char.character_id in mentioned_ids
                ]
    
            # Filter objects based on the filter flags
            filtered_objects = []
            for obj in image_analysis.objects:
                # Keep if it's a background element and we're not filtering backgrounds
                if obj.object_id.startswith('bg') and not filter_backgrounds:
                    filtered_objects.append(obj)
                # Keep if it's a regular object and we're not filtering objects
                elif obj.object_id.startswith('obj') and not filter_objects:
                    filtered_objects.append(obj)
                # Otherwise, keep only if it's mentioned in the story
                elif obj.object_id in mentioned_ids:
                    filtered_objects.append(obj)
                # If it doesn't start with 'bg' or 'obj', just keep it
                elif not (obj.object_id.startswith('bg') or obj.object_id.startswith('obj')):
                    filtered_objects.append(obj)
    
            image_analysis.objects = filtered_objects
    
        return filtered_cot
    
    
    
    @staticmethod
    def _extract_mentioned_ids(story: Story) -> set:
        """
        Extract all character and object IDs mentioned in the story text.
        
        Args:
            story: Story object containing text with element references
            
        Returns:
            Set of all mentioned character and object IDs
        """
        mentioned_ids = set()
    
        # Regular expressions to find character and object references in <gdo> and similar tags
        id_pattern = r'<gd[oal]\s+([^>]+)>'
    
        # Check each image's text for mentioned IDs
        for image in story.images:
            # Find all matches in the current image text
            matches = re.findall(id_pattern, image.text)
            for match in matches:
                # Extract the ID - it might be the entire tag content or just part of it
                # In the cases like <gdo char6>, the ID is "char6"
                # In more complex like <gdo char6 obj3>, we need to split the match and check each part
                match_parts = match.split()
                for part in match_parts:
                    if part.startswith('char') or part.startswith('obj') or part.startswith('bg'):
                        mentioned_ids.add(part)
                        
            

    
        return mentioned_ids