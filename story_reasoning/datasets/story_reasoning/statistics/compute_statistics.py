import glob
import json
import os
import re
import statistics
from collections import defaultdict
from typing import Dict, List
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

import matplotlib.pyplot as plt
import numpy as np
import spacy
from tqdm import tqdm

from story_reasoning.models.story_reasoning.story_reasoning_util import StoryReasoningUtil
from story_reasoning.models.story_reasoning.narrative_phase import NarrativePhase

# Constants for chart formatting
SMALL_SIZE = 16    # For tick labels
MEDIUM_SIZE = 16   # For axis labels
LARGE_SIZE = 16    # For titles
ANNOTATION_SIZE = 12  # For value annotations


def setup_matplotlib_style():
    """Set up consistent matplotlib styling for all charts"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rc('font', size=MEDIUM_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)


def load_spacy_model():
    """Load the spaCy model for NLP tasks"""
    try:
        return spacy.load("en_core_web_sm")
    except:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")


def count_grounding_tags(story_content):
    """Count different types of grounding tags in the story"""
    # Count character references (gdo tags for characters)
    char_refs = re.findall(r'<gdo char\d+[^>]*>(.*?)</gdo>', story_content, re.DOTALL)

    # Define pronoun categories
    subject_pronouns = ['he', 'she', 'they', 'I', 'we', 'you']
    possessive_pronouns = ['his', 'her', 'their', 'my', 'our', 'your']
    object_pronouns = ['him', 'them', 'me', 'us']
    all_pronouns = subject_pronouns + possessive_pronouns + object_pronouns

    pronoun_counts = {pronoun: 0 for pronoun in all_pronouns}
    pronoun_grounded = {pronoun: 0 for pronoun in all_pronouns}

    for ref in char_refs:
        ref_lower = ref.lower()
        for pronoun in all_pronouns:
            if ref_lower == pronoun:
                pronoun_counts[pronoun] += 1
                pronoun_grounded[pronoun] += 1

    # Count specifically by pronoun type
    subject_pronoun_refs = sum(pronoun_counts[p] for p in subject_pronouns)
    possessive_pronoun_refs = sum(pronoun_counts[p] for p in possessive_pronouns)
    pronoun_refs = sum(pronoun_counts.values())

    # Count other reference types by leveraging regex once
    obj_refs = re.findall(r'<gdo obj\d+[^>]*>(.*?)</gdo>', story_content, re.DOTALL)
    action_refs = re.findall(r'<gda[^>]*>(.*?)</gda>', story_content, re.DOTALL)
    setting_refs = re.findall(r'<gdl[^>]*>(.*?)</gdl>', story_content, re.DOTALL)

    return {
        'character_refs': len(char_refs),
        'pronoun_refs': pronoun_refs,
        'subject_pronoun_refs': subject_pronoun_refs,
        'possessive_pronoun_refs': possessive_pronoun_refs,
        'pronoun_counts': pronoun_counts,
        'pronoun_grounded': pronoun_grounded,
        'subject_pronouns': subject_pronouns,
        'possessive_pronouns': possessive_pronouns,
        'object_pronouns': object_pronouns,
        'object_refs': len(obj_refs),
        'action_refs': len(action_refs),
        'setting_refs': len(setting_refs),
        'total_refs': len(char_refs) + len(obj_refs) + len(action_refs) + len(setting_refs)
    }


def identify_ungrounded_entities(story_content, nlp):
    """Identify and count ungrounded pronouns and person proper nouns in the story text"""
    # Remove all XML tags to get the clean text
    clean_text = StoryReasoningUtil.strip_story_tags(story_content)

    # Parse the text with spaCy
    doc = nlp(clean_text)

    # Define pronoun categories
    subject_pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they']
    possessive_pronouns = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
    object_pronouns = ['me', 'you', 'him', 'her', 'it', 'us', 'them']
    all_pronouns = subject_pronouns + possessive_pronouns + object_pronouns

    # Find all pronouns in the clean text
    all_pronouns_dict = {pronoun: 0 for pronoun in all_pronouns}

    for token in doc:
        if token.pos_ == "PRON" and token.text.lower() in all_pronouns:
            all_pronouns_dict[token.text.lower()] += 1

    # Calculate pronoun counts by type
    all_pronouns_count = sum(all_pronouns_dict.values())
    subject_pronouns_count = sum(all_pronouns_dict[p] for p in subject_pronouns)
    possessive_pronouns_count = sum(all_pronouns_dict[p] for p in possessive_pronouns)

    # Find person proper nouns
    person_proper_nouns = [
        token.text.lower() for token in doc
        if token.pos_ == "PROPN" and len(token.text) > 2 and not token.is_stop
           and (token.ent_type_ == "PERSON" or
                (token.text[0].isupper() and token.ent_type_ not in ["LOC", "GPE", "ORG", "FAC", "DATE", "TIME", "MONEY", "QUANTITY"]))
    ]

    # Find all grounded pronouns
    grounded_pronoun_matches = re.findall(r'<gdo char\d+[^>]*>(.*?)</gdo>', story_content, re.DOTALL)
    grounded_pronouns_dict = {pronoun: 0 for pronoun in all_pronouns}

    for match in grounded_pronoun_matches:
        match_lower = match.lower()
        if match_lower in all_pronouns:
            grounded_pronouns_dict[match_lower] += 1

    # Calculate pronoun groundings by type
    grounded_pronouns = sum(grounded_pronouns_dict.values())
    grounded_subject_pronouns = sum(grounded_pronouns_dict[p] for p in subject_pronouns)
    grounded_possessive_pronouns = sum(grounded_pronouns_dict[p] for p in possessive_pronouns)

    # Find all grounded proper nouns from character tags
    grounded_char_matches = re.findall(r'<gdo char\d+[^>]*>(.*?)</gdo>', story_content, re.DOTALL)
    grounded_proper_nouns = []

    for match in grounded_char_matches:
        match_doc = nlp(match)
        for word in match_doc:
            if word.pos_ == "PROPN" and len(word.text) > 2 and not word.is_stop:
                grounded_proper_nouns.append(word.text.lower())

    # Calculate ungrounded entities
    ungrounded_pronouns_dict = {
        pronoun: max(0, all_pronouns_dict[pronoun] - grounded_pronouns_dict[pronoun])
        for pronoun in all_pronouns
    }

    ungrounded_pronouns = sum(ungrounded_pronouns_dict.values())
    ungrounded_subject_pronouns = sum(ungrounded_pronouns_dict[p] for p in subject_pronouns)
    ungrounded_possessive_pronouns = sum(ungrounded_pronouns_dict[p] for p in possessive_pronouns)

    ungrounded_proper_nouns = [
        pn for pn in person_proper_nouns
        if pn not in [pn.lower() for pn in grounded_proper_nouns]
    ]

    # Calculate ungrounded percentage for each pronoun type
    ungrounded_pronoun_pct = {
        pronoun: (ungrounded_pronouns_dict[pronoun] / all_pronouns_dict[pronoun] * 100)
        if all_pronouns_dict[pronoun] > 0 else 0
        for pronoun in all_pronouns
    }

    ungrounded_subject_pronoun_pct = {
        pronoun: (ungrounded_pronouns_dict[pronoun] / all_pronouns_dict[pronoun] * 100)
        if all_pronouns_dict[pronoun] > 0 else 0
        for pronoun in subject_pronouns
    }

    ungrounded_possessive_pronoun_pct = {
        pronoun: (ungrounded_pronouns_dict[pronoun] / all_pronouns_dict[pronoun] * 100)
        if all_pronouns_dict[pronoun] > 0 else 0
        for pronoun in possessive_pronouns
    }

    return {
        'total_pronouns': all_pronouns_count,
        'total_subject_pronouns': subject_pronouns_count,
        'total_possessive_pronouns': possessive_pronouns_count,
        'pronoun_counts': all_pronouns_dict,
        'grounded_pronouns': grounded_pronouns,
        'grounded_subject_pronouns': grounded_subject_pronouns,
        'grounded_possessive_pronouns': grounded_possessive_pronouns,
        'grounded_pronouns_dict': grounded_pronouns_dict,
        'ungrounded_pronouns': ungrounded_pronouns,
        'ungrounded_subject_pronouns': ungrounded_subject_pronouns,
        'ungrounded_possessive_pronouns': ungrounded_possessive_pronouns,
        'ungrounded_pronouns_dict': ungrounded_pronouns_dict,
        'ungrounded_pronoun_pct': ungrounded_pronoun_pct,
        'ungrounded_subject_pronoun_pct': ungrounded_subject_pronoun_pct,
        'ungrounded_possessive_pronoun_pct': ungrounded_possessive_pronoun_pct,
        'subject_pronouns': subject_pronouns,
        'possessive_pronouns': possessive_pronouns,
        'ungrounded_pronoun_list': list(ungrounded_pronouns_dict.keys()),
        'total_proper_nouns': len(person_proper_nouns),
        'grounded_proper_nouns': len(grounded_proper_nouns),
        'ungrounded_proper_nouns': len(ungrounded_proper_nouns),
        'ungrounded_proper_noun_list': ungrounded_proper_nouns
    }


def initialize_results_structure():
    """Initialize the data structure for storing analysis results"""
    return {
        'story_count': 0,
        'frames_per_story': [],
        'unique_entities': {
            'characters': [],
            'objects': [],
            'backgrounds': [],
            'landmarks': []
        },
        'cross_frame_entities': {
            'characters': [],
            'objects': [],
            'characters_frames_distribution': defaultdict(int),
            'objects_frames_distribution': defaultdict(int)
        },
        'grounding_stats': {
            'entity_refs': [],
            'character_refs': [],
            'object_refs': [],
            'setting_refs': [],
            'action_refs': [],
            'pronoun_percentage': [],
            'subject_pronoun_percentage': [],
            'possessive_pronoun_percentage': [],
            'pronoun_counts': [],
            'pronoun_grounded': []
        },
        'ungrounded_stats': {
            'pronouns_by_story': [],
            'subject_pronouns_by_story': [],
            'possessive_pronouns_by_story': [],
            'nouns_by_story': [],
            'pronouns_by_story_length': [],
            'subject_pronouns_by_story_length': [],
            'possessive_pronouns_by_story_length': [],
            'nouns_by_story_length': [],
            'ungrounded_pronouns': defaultdict(list),
            'ungrounded_subject_pronouns': defaultdict(list),
            'ungrounded_possessive_pronouns': defaultdict(list)
        },
        'structure_stats': {
            'characters_per_image': [],
            'objects_per_image': [],
            'settings_per_image': [],
            'complete_narratives': 0,
            'narrative_phases': defaultdict(set),
            'stories_by_phase': defaultdict(int)
        },
        'story_stats': {
            'word_counts': [],
            'distinct_characters': [],
            'character_actions': [],
            'char_count_by_length': [],
            'obj_count_by_length': []
        },
        'all_characters': [],
        'all_objects': []
    }


def process_story_metadata(story_id: str, dataset_path: str) -> List[str]:
    """Process the metadata file for a story and return frames"""
    metadata_file = os.path.join(dataset_path, f"{story_id}_metadata.txt")
    frames = []

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            frames = f.read().strip().split('\n')

    return frames


def process_story_cot(story_id: str, dataset_path: str, frames: List[str], results: Dict) -> None:
    """Process the chain of thought file for a story and update results"""
    cot_file = os.path.join(dataset_path, f"{story_id}_cot.txt")

    if not os.path.exists(cot_file):
        return

    with open(cot_file, 'r', encoding='utf-8') as f:
        cot_content = f.read()

    # Use StoryReasoningUtil to parse the CoT content
    cot = StoryReasoningUtil.parse_cot(cot_content)

    if not cot or not cot.images:
        return

    # Extract entity information
    process_entities_from_cot(cot, frames, results, story_id)


def process_entities_from_cot(cot, frames, results, story_id):
    """Extract and process entity information from CoT"""
    all_characters = []
    all_objects = []
    all_settings = []

    for image_analysis in cot.images:
        all_characters.extend(image_analysis.characters)
        all_objects.extend(image_analysis.objects)
        all_settings.extend(image_analysis.settings)

        # Track per-image stats
        results['structure_stats']['characters_per_image'].append(len(image_analysis.characters))
        results['structure_stats']['objects_per_image'].append(len(image_analysis.objects))
        results['structure_stats']['settings_per_image'].append(len(image_analysis.settings))

    # Store for visualization
    results['all_characters'].extend(all_characters)
    results['all_objects'].extend(all_objects)

    # Process unique entities
    process_unique_entities(all_characters, all_objects, all_settings, results)

    # Process cross-frame entities
    process_cross_frame_entities(cot.images, results)

    # Process narrative structure
    process_narrative_structure(cot, story_id, results)


def process_unique_entities(all_characters, all_objects, all_settings, results):
    """Extract and count unique entities"""
    unique_chars = {char.character_id for char in all_characters}
    unique_objs = {obj.object_id for obj in all_objects if not obj.object_id.startswith('bg')}
    unique_bgs = {obj.object_id for obj in all_objects if obj.object_id.startswith('bg')}
    unique_landmarks = sum(1 for setting in all_settings if 'landmark' in setting.setting_element.value.lower())

    results['unique_entities']['characters'].append(len(unique_chars))
    results['unique_entities']['objects'].append(len(unique_objs))
    results['unique_entities']['backgrounds'].append(len(unique_bgs))
    results['unique_entities']['landmarks'].append(unique_landmarks)


def process_cross_frame_entities(images, results):
    """Process entities that appear across multiple frames"""
    char_frames = defaultdict(set)
    obj_frames = defaultdict(set)

    # Count entities by frame
    for i, image_analysis in enumerate(images):
        frame_chars = {char.character_id for char in image_analysis.characters}
        frame_objs = {obj.object_id for obj in image_analysis.objects if not obj.object_id.startswith('bg')}

        for char_id in frame_chars:
            char_frames[char_id].add(i)
        for obj_id in frame_objs:
            obj_frames[obj_id].add(i)

    # Calculate percentage of entities in multiple frames
    unique_chars = set(char_frames.keys())
    unique_objs = set(obj_frames.keys())

    multi_frame_chars = sum(1 for char_id, frames in char_frames.items() if len(frames) >= 2)
    multi_frame_objs = sum(1 for obj_id, frames in obj_frames.items() if len(frames) >= 2)

    if unique_chars:  # Avoid division by zero
        results['cross_frame_entities']['characters'].append(multi_frame_chars / len(unique_chars) * 100)
    if unique_objs:  # Avoid division by zero
        results['cross_frame_entities']['objects'].append(multi_frame_objs / len(unique_objs) * 100)

    # Calculate frame distribution
    calculate_frame_distribution(char_frames, obj_frames, unique_chars, unique_objs, len(images), results)


def calculate_frame_distribution(char_frames, obj_frames, unique_chars, unique_objs, max_frames, results):
    """Calculate distribution of entities across N or more frames"""
    for n in range(1, max_frames + 1):
        chars_in_n_or_more = sum(1 for char_id, frames in char_frames.items() if len(frames) >= n)
        objs_in_n_or_more = sum(1 for obj_id, frames in obj_frames.items() if len(frames) >= n)

        if unique_chars:
            char_pct = (chars_in_n_or_more / len(unique_chars)) * 100
            results['cross_frame_entities']['characters_frames_distribution'][n] += char_pct

        if unique_objs:
            obj_pct = (objs_in_n_or_more / len(unique_objs)) * 100
            results['cross_frame_entities']['objects_frames_distribution'][n] += obj_pct


def process_narrative_structure(cot, story_id, results):
    """Process narrative structure information"""
    if not cot.narrative_structure or not cot.narrative_structure.phases:
        return

    if len(cot.narrative_structure.phases) == 5:  # All 5 phases present
        results['structure_stats']['complete_narratives'] += 1

    # Track phases per story
    story_phases = set()
    for phase in cot.narrative_structure.phases:
        if 'Narrative Phase' in phase and hasattr(phase['Narrative Phase'], 'value'):
            phase_name = phase['Narrative Phase'].value
            results['structure_stats']['narrative_phases'][phase_name].add(story_id)
            story_phases.add(phase_name)

    # Count each phase only once per story
    for phase in story_phases:
        results['structure_stats']['stories_by_phase'][phase] += 1


def process_story_text(story_id, dataset_path, results, nlp):
    """Process the story text file for grounding and ungrounded stats"""
    story_file = os.path.join(dataset_path, f"{story_id}_story.txt")

    if not os.path.exists(story_file):
        return

    with open(story_file, 'r', encoding='utf-8') as f:
        story_content = f.read()
        
    # Count grounding tags
    tag_counts = count_grounding_tags(story_content)
    update_grounding_stats(tag_counts, results)
    

    # Identify ungrounded entities
    ungrounded_stats = identify_ungrounded_entities(story_content, nlp)
    update_ungrounded_stats(ungrounded_stats, results)
    
    # Process story statistics
    process_story_statistics(story_content, ungrounded_stats, results)
    


def update_grounding_stats(tag_counts, results):
    """Update results with grounding statistics"""
    results['grounding_stats']['entity_refs'].append(tag_counts['total_refs'])
    results['grounding_stats']['character_refs'].append(tag_counts['character_refs'])
    results['grounding_stats']['object_refs'].append(tag_counts['object_refs'])
    results['grounding_stats']['setting_refs'].append(tag_counts['setting_refs'])
    results['grounding_stats']['action_refs'].append(tag_counts['action_refs'])
    results['grounding_stats']['pronoun_counts'].append(tag_counts['pronoun_counts'])
    results['grounding_stats']['pronoun_grounded'].append(tag_counts['pronoun_grounded'])

    # Calculate pronoun percentage
    if tag_counts['character_refs'] > 0:
        results['grounding_stats']['pronoun_percentage'].append(
            tag_counts['pronoun_refs'] / tag_counts['character_refs'] * 100)
        results['grounding_stats']['subject_pronoun_percentage'].append(
            tag_counts['subject_pronoun_refs'] / tag_counts['character_refs'] * 100)
        results['grounding_stats']['possessive_pronoun_percentage'].append(
            tag_counts['possessive_pronoun_refs'] / tag_counts['character_refs'] * 100)


def update_ungrounded_stats(ungrounded_stats, results):
    """Update results with ungrounded entity statistics"""
    # Calculate percentages of ungrounded entities
    if ungrounded_stats['total_pronouns'] > 0:
        ungrounded_pronoun_pct = (ungrounded_stats['ungrounded_pronouns'] / ungrounded_stats['total_pronouns']) * 100
        results['ungrounded_stats']['pronouns_by_story'].append(ungrounded_pronoun_pct)

        # Subject pronouns percentage
        if ungrounded_stats['total_subject_pronouns'] > 0:
            ungrounded_subject_pct = (ungrounded_stats['ungrounded_subject_pronouns'] / ungrounded_stats['total_subject_pronouns']) * 100
            results['ungrounded_stats']['subject_pronouns_by_story'].append(ungrounded_subject_pct)

        # Possessive pronouns percentage
        if ungrounded_stats['total_possessive_pronouns'] > 0:
            ungrounded_possessive_pct = (ungrounded_stats['ungrounded_possessive_pronouns'] / ungrounded_stats['total_possessive_pronouns']) * 100
            results['ungrounded_stats']['possessive_pronouns_by_story'].append(ungrounded_possessive_pct)

        # Store ungrounded percentages for each pronoun
        for pronoun, pct in ungrounded_stats['ungrounded_pronoun_pct'].items():
            if ungrounded_stats['pronoun_counts'][pronoun] > 0:
                results['ungrounded_stats']['ungrounded_pronouns'][pronoun].append(pct)

        # Store ungrounded percentages for subject and possessive pronouns
        for pronoun in ungrounded_stats['subject_pronouns']:
            if ungrounded_stats['pronoun_counts'][pronoun] > 0:
                results['ungrounded_stats']['ungrounded_subject_pronouns'][pronoun].append(
                    ungrounded_stats['ungrounded_subject_pronoun_pct'][pronoun])

        for pronoun in ungrounded_stats['possessive_pronouns']:
            if ungrounded_stats['pronoun_counts'][pronoun] > 0:
                results['ungrounded_stats']['ungrounded_possessive_pronouns'][pronoun].append(
                    ungrounded_stats['ungrounded_possessive_pronoun_pct'][pronoun])

    if ungrounded_stats['total_proper_nouns'] > 0:
        ungrounded_proper_noun_pct = (ungrounded_stats['ungrounded_proper_nouns'] / ungrounded_stats['total_proper_nouns']) * 100
        results['ungrounded_stats']['nouns_by_story'].append(ungrounded_proper_noun_pct)


def process_story_statistics(story_content, ungrounded_stats, results):
    """Process general statistics about the story"""
    # Get clean text by removing tags
    clean_text = StoryReasoningUtil.strip_story_tags(story_content)

    # Count words
    words = re.findall(r'\b\w+\b', clean_text)
    word_count = len(words)
    results['story_stats']['word_counts'].append(word_count)

    # Store pronoun data with story length
    store_pronoun_data_by_length(word_count, ungrounded_stats, results)

    # Count distinct entities in story
    count_distinct_entities(story_content, word_count, results)


def store_pronoun_data_by_length(word_count, ungrounded_stats, results):
    """Store pronoun statistics relative to story length"""
    local_vars = locals()

    # Store ungrounded pronouns by story length
    if 'ungrounded_pronoun_pct' in local_vars and ungrounded_stats['total_pronouns'] > 0:
        ungrounded_pronoun_pct = (ungrounded_stats['ungrounded_pronouns'] / ungrounded_stats['total_pronouns']) * 100
        results['ungrounded_stats']['pronouns_by_story_length'].append((word_count, ungrounded_pronoun_pct))

    # Store ungrounded subject pronouns by story length
    if ungrounded_stats['total_subject_pronouns'] > 0:
        ungrounded_subject_pct = (ungrounded_stats['ungrounded_subject_pronouns'] / ungrounded_stats['total_subject_pronouns']) * 100
        results['ungrounded_stats']['subject_pronouns_by_story_length'].append((word_count, ungrounded_subject_pct))

    # Store ungrounded possessive pronouns by story length
    if ungrounded_stats['total_possessive_pronouns'] > 0:
        ungrounded_possessive_pct = (ungrounded_stats['ungrounded_possessive_pronouns'] / ungrounded_stats['total_possessive_pronouns']) * 100
        results['ungrounded_stats']['possessive_pronouns_by_story_length'].append((word_count, ungrounded_possessive_pct))

    # Store ungrounded proper nouns by story length
    if ungrounded_stats['total_proper_nouns'] > 0:
        ungrounded_proper_noun_pct = (ungrounded_stats['ungrounded_proper_nouns'] / ungrounded_stats['total_proper_nouns']) * 100
        results['ungrounded_stats']['nouns_by_story_length'].append((word_count, ungrounded_proper_noun_pct))


def count_distinct_entities(story_content, word_count, results):
    """Count distinct entities in the story"""
    # Count distinct characters
    char_ids_in_story = set(re.findall(r'<gdo char(\d+)', story_content))
    char_count = len(char_ids_in_story)
    results['story_stats']['distinct_characters'].append(char_count)

    # Count distinct objects
    obj_ids_in_story = set(re.findall(r'<gdo obj(\d+)', story_content))
    obj_count = len(obj_ids_in_story)

    # Store character and object counts by story length
    results['story_stats']['char_count_by_length'].append((word_count, char_count))
    results['story_stats']['obj_count_by_length'].append((word_count, obj_count))

    # Count character actions
    actions = re.findall(r'<gda char\d+[^>]*>(.*?)</gda>', story_content, re.DOTALL)
    results['story_stats']['character_actions'].append(len(actions))


def analyze_dataset(dataset_path):
    """Analyze the complete dataset and calculate statistics"""
    nlp = load_spacy_model()
    results = initialize_results_structure()

    # Get all story IDs
    story_ids = set()
    for file in glob.glob(os.path.join(dataset_path, "*_metadata.txt")):
        story_id = os.path.basename(file).split("_")[0]
        story_ids.add(story_id)
        
    results['story_count'] = len(story_ids)

    # Process each story
    for story_id in tqdm(story_ids, desc="Processing stories"):
        # Process metadata file to get frames
        frames = process_story_metadata(story_id, dataset_path)
        if frames:
            results['frames_per_story'].append(len(frames))
            
        # Process CoT file to get structured analysis
        process_story_cot(story_id, dataset_path, frames, results)

        # Process story file for grounding stats
        process_story_text(story_id, dataset_path, results, nlp)
        
    # Calculate averages for frame distributions
    normalize_frame_distributions(results)

    # Calculate final statistics
    stats = calculate_final_statistics(results)

    # Create visualization data
    visualize_data = {
        'raw_results': results,
        'stats': stats
    }

    return stats, visualize_data


def normalize_frame_distributions(results):
    """Normalize frame distribution percentages by story count"""
    story_count = results['story_count']
    for n in results['cross_frame_entities']['characters_frames_distribution']:
        results['cross_frame_entities']['characters_frames_distribution'][n] /= story_count
    for n in results['cross_frame_entities']['objects_frames_distribution']:
        results['cross_frame_entities']['objects_frames_distribution'][n] /= story_count


def calculate_final_statistics(results):
    """Calculate final aggregate statistics from results"""
    # Calculate average ungrounded percentage for each pronoun type
    avg_ungrounded_by_pronoun = calculate_avg_ungrounded_by_type(
        results['ungrounded_stats']['ungrounded_pronouns'])

    avg_ungrounded_by_subject_pronoun = calculate_avg_ungrounded_by_type(
        results['ungrounded_stats']['ungrounded_subject_pronouns'])

    avg_ungrounded_by_possessive_pronoun = calculate_avg_ungrounded_by_type(
        results['ungrounded_stats']['ungrounded_possessive_pronouns'])

    # Calculate final statistics
    return {
        'total_stories': {
            'value': results['story_count'],
            'description': 'Total number of stories in the dataset'
        },
        'avg_frames_per_story': {
            'value': safe_mean(results['frames_per_story']),
            'description': 'Average number of frames/images per story'
        },
        'max_frames_per_story': {
            'value': max(results['frames_per_story']) if results['frames_per_story'] else 0,
            'description': 'Maximum number of frames/images in any story'
        },
        'pct_chars_multi_frame': {
            'value': safe_mean(results['cross_frame_entities']['characters']),
            'description': 'Percentage of characters that appear in multiple frames, showing identity preservation'
        },
        'pct_objs_multi_frame': {
            'value': safe_mean(results['cross_frame_entities']['objects']),
            'description': 'Percentage of objects that appear in multiple frames, showing identity preservation'
        },
        'avg_entity_refs': {
            'value': safe_mean(results['grounding_stats']['entity_refs']),
            'description': 'Average number of entity references per story (sum of character, object, setting, and action references)'
        },
        'avg_char_refs': {
            'value': safe_mean(results['grounding_stats']['character_refs']),
            'description': 'Average number of character references per story'
        },
        'avg_obj_refs': {
            'value': safe_mean(results['grounding_stats']['object_refs']),
            'description': 'Average number of object references per story'
        },
        'avg_setting_refs': {
            'value': safe_mean(results['grounding_stats']['setting_refs']),
            'description': 'Average number of setting/location references per story'
        },
        'avg_action_refs': {
            'value': safe_mean(results['grounding_stats']['action_refs']),
            'description': 'Average number of action references per story'
        },
        'avg_pronoun_pct': {
            'value': safe_mean(results['grounding_stats']['pronoun_percentage']),
            'description': 'Average percentage of character references that are pronouns (he, she, they, etc.)'
        },
        'avg_subject_pronoun_pct': {
            'value': safe_mean(results['grounding_stats']['subject_pronoun_percentage']),
            'description': 'Average percentage of character references that are subject pronouns (he, she, they, etc.)'
        },
        'avg_possessive_pronoun_pct': {
            'value': safe_mean(results['grounding_stats']['possessive_pronoun_percentage']),
            'description': 'Average percentage of character references that are possessive pronouns (his, her, their, etc.)'
        },
        'avg_ungrounded_pronouns_pct': {
            'value': safe_mean(results['ungrounded_stats']['pronouns_by_story']),
            'description': 'Average percentage of pronouns that are not grounded to specific characters'
        },
        'avg_ungrounded_subject_pronouns_pct': {
            'value': safe_mean(results['ungrounded_stats']['subject_pronouns_by_story']),
            'description': 'Average percentage of subject pronouns that are not grounded to specific characters'
        },
        'avg_ungrounded_possessive_pronouns_pct': {
            'value': safe_mean(results['ungrounded_stats']['possessive_pronouns_by_story']),
            'description': 'Average percentage of possessive pronouns that are not grounded to specific characters'
        },
        'avg_ungrounded_nouns_pct': {
            'value': safe_mean(results['ungrounded_stats']['nouns_by_story']),
            'description': 'Average percentage of proper nouns that are not grounded to specific objects'
        },
        'ungrounded_by_pronoun': {
            'value': avg_ungrounded_by_pronoun,
            'description': 'Average percentage of ungrounded occurrences for each pronoun'
        },
        'ungrounded_by_subject_pronoun': {
            'value': avg_ungrounded_by_subject_pronoun,
            'description': 'Average percentage of ungrounded occurrences for each subject pronoun'
        },
        'ungrounded_by_possessive_pronoun': {
            'value': avg_ungrounded_by_possessive_pronoun,
            'description': 'Average percentage of ungrounded occurrences for each possessive pronoun'
        },
        'avg_chars_per_image': {
            'value': safe_mean(results['structure_stats']['characters_per_image']),
            'description': 'Average number of characters per image in the structured analysis'
        },
        'avg_objs_per_image': {
            'value': safe_mean(results['structure_stats']['objects_per_image']),
            'description': 'Average number of objects per image in the structured analysis'
        },
        'avg_settings_per_image': {
            'value': safe_mean(results['structure_stats']['settings_per_image']),
            'description': 'Average number of setting elements per image in the structured analysis'
        },
        'pct_complete_narratives': {
            'value': (results['structure_stats']['complete_narratives'] / results['story_count'] * 100) if results['story_count'] else 0,
            'description': 'Percentage of stories containing all five narrative phases (Introduction, Development, Conflict, Turning Point, Conclusion)'
        },
        'avg_words_per_story': {
            'value': safe_mean(results['story_stats']['word_counts']),
            'description': 'Average number of words per story'
        },
        'avg_distinct_chars': {
            'value': safe_mean(results['story_stats']['distinct_characters']),
            'description': 'Average number of distinct characters appearing in each story'
        },
        'avg_char_actions': {
            'value': safe_mean(results['story_stats']['character_actions']),
            'description': 'Average number of character actions per story'
        }
    }


def calculate_avg_ungrounded_by_type(pronoun_data):
    """Calculate average ungrounded percentage for each pronoun"""
    avg_ungrounded = {}
    for pronoun, percentages in pronoun_data.items():
        if percentages:
            avg_ungrounded[pronoun] = statistics.mean(percentages)
        else:
            avg_ungrounded[pronoun] = 0
    return avg_ungrounded


def safe_mean(data_list):
    """Safely calculate mean, returning 0 for empty lists"""
    return statistics.mean(data_list) if data_list else 0


# Chart creation functions
def create_frames_distribution_chart(results, output_dir):
    """Create chart showing distribution of frames per story"""
    plt.figure(figsize=(10, 6))
    plt.hist(results['frames_per_story'],
             bins=range(min(results['frames_per_story']), max(results['frames_per_story']) + 2),
             alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Number of Frames', fontsize=MEDIUM_SIZE)
    plt.ylabel('Number of Stories', fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'frames_distribution.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


def create_cross_frame_consistency_chart(results, output_dir):
    """Create chart showing entity persistence across frames"""
    frame_counts = sorted(results['cross_frame_entities']['characters_frames_distribution'].keys())
    char_percentages = [results['cross_frame_entities']['characters_frames_distribution'][n] for n in frame_counts]
    obj_percentages = [results['cross_frame_entities']['objects_frames_distribution'][n] for n in frame_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(frame_counts, char_percentages, 'o-', color='#9b59b6', linewidth=2, markersize=8, label='Characters')
    plt.plot(frame_counts, obj_percentages, 's-', color='#f1c40f', linewidth=2, markersize=8, label='Objects')

    plt.xlabel('Number of Frames (N)', fontsize=MEDIUM_SIZE)
    plt.ylabel('Percentage Appearing in ≥ N Frames', fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=MEDIUM_SIZE)
    plt.xticks(frame_counts, fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.ylim(0, 105)

    # Add percentage labels
    for i, (x, y) in enumerate(zip(frame_counts, char_percentages)):
        plt.annotate(f'{y:.1f}%', (x, y + 3), ha='center', fontsize=ANNOTATION_SIZE, color='#9b59b6', fontweight='bold')

    for i, (x, y) in enumerate(zip(frame_counts, obj_percentages)):
        plt.annotate(f'{y:.1f}%', (x, y - 3), ha='center', fontsize=ANNOTATION_SIZE, color='#f1c40f', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_frame_consistency.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


def create_grounding_stats_chart(stats, output_dir):
    """Create chart showing grounding statistics by reference type"""
    ref_types = ['Character', 'Character Action', 'Object', 'Setting']
    ref_counts = [
        stats['avg_char_refs']['value'],
        stats['avg_action_refs']['value'],
        stats['avg_obj_refs']['value'],
        stats['avg_setting_refs']['value']
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(ref_types, ref_counts, color=['#16a085', '#c0392b', '#d35400', '#8e44ad'], alpha=0.8,
                   edgecolor='black')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold')

    plt.xlabel('Reference Type', fontsize=MEDIUM_SIZE)
    plt.ylabel('Average References per Story', fontsize=MEDIUM_SIZE)
    plt.ylim(0, max(ref_counts) * 1.2)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'grounding_stats.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


def create_scatter_chart(data_points, output_dir, filename, title, x_label, y_label, color):
    """Create a scatter chart with trend line"""
    if not data_points:
        return None

    # Sort by word count
    sorted_data = sorted(data_points, key=lambda x: x[0])

    # Get x and y values
    x_values = [item[0] for item in sorted_data]
    y_values = [item[1] for item in sorted_data]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.6, c=color, edgecolor='black', s=80)

    # Add trend line if enough data points
    if len(x_values) > 1:
        z = np.polyfit(x_values, y_values, 1)
        p = np.poly1d(z)
        plt.plot(x_values, p(x_values), "--", alpha=0.7, linewidth=2, color=color)

    plt.xlabel(x_label, fontsize=MEDIUM_SIZE)
    plt.ylabel(y_label, fontsize=MEDIUM_SIZE)
    plt.title(title, fontsize=LARGE_SIZE)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


def create_ungrounded_subject_pronouns_chart(results, output_dir):
    """Create chart showing relationship between story length and ungrounded subject pronouns"""
    return create_scatter_chart(
        results['ungrounded_stats']['subject_pronouns_by_story_length'],
        output_dir,
        'ungrounded_subject_pronouns_by_length.pdf',
        'Ungrounded Subject Pronouns vs Story Length',
        'Story Length (words)',
        'Ungrounded Subject Pronouns (%)',
        '#e74c3c'
    )


def create_ungrounded_possessive_pronouns_chart(results, output_dir):
    """Create chart showing relationship between story length and ungrounded possessive pronouns"""
    return create_scatter_chart(
        results['ungrounded_stats']['possessive_pronouns_by_story_length'],
        output_dir,
        'ungrounded_possessive_pronouns_by_length.pdf',
        'Ungrounded Possessive Pronouns vs Story Length',
        'Story Length (words)',
        'Ungrounded Possessive Pronouns (%)',
        '#2980b9'
    )


def create_ungrounded_nouns_chart(results, output_dir):
    """Create chart showing relationship between story length and ungrounded person nouns"""
    return create_scatter_chart(
        results['ungrounded_stats']['nouns_by_story_length'],
        output_dir,
        'ungrounded_person_nouns_by_length.pdf',
        'Ungrounded Person Nouns vs Story Length',
        'Story Length (words)',
        'Ungrounded Nouns (%)',
        '#3498db'
    )


def create_entities_by_story_length_chart(results, output_dir):
    """Create chart showing relationship between story length and entity counts"""
    if not (results['story_stats']['char_count_by_length'] and results['story_stats']['obj_count_by_length']):
        return None

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sort by word count
    char_by_length = sorted(results['story_stats']['char_count_by_length'], key=lambda x: x[0])
    obj_by_length = sorted(results['story_stats']['obj_count_by_length'], key=lambda x: x[0])

    # Get x and y values
    char_x = [item[0] for item in char_by_length]
    char_y = [item[1] for item in char_by_length]

    obj_x = [item[0] for item in obj_by_length]
    obj_y = [item[1] for item in obj_by_length]

    # Character scatter plot
    ax1.scatter(char_x, char_y, alpha=0.6, c='#9b59b6', edgecolor='black', s=80)

    # Add trend line for characters
    if len(char_x) > 1:
        z = np.polyfit(char_x, char_y, 1)
        p = np.poly1d(z)
        ax1.plot(char_x, p(char_x), "r--", alpha=0.7, linewidth=2)

    ax1.set_xlabel('Story Length (words)', fontsize=MEDIUM_SIZE)
    ax1.set_ylabel('Number of Characters', fontsize=MEDIUM_SIZE)
    ax1.set_title('Characters vs Story Length', fontsize=LARGE_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)

    # Object scatter plot
    ax2.scatter(obj_x, obj_y, alpha=0.6, c='#f1c40f', edgecolor='black', s=80)

    # Add trend line for objects
    if len(obj_x) > 1:
        z = np.polyfit(obj_x, obj_y, 1)
        p = np.poly1d(z)
        ax2.plot(obj_x, p(obj_x), "b--", alpha=0.7, linewidth=2)

    ax2.set_xlabel('Story Length (words)', fontsize=MEDIUM_SIZE)
    ax2.set_ylabel('Number of Objects', fontsize=MEDIUM_SIZE)
    ax2.set_title('Objects vs Story Length', fontsize=LARGE_SIZE)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'entities_by_story_length.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


def create_narrative_phase_chart(results, output_dir):
    """Create chart showing distribution of narrative phases in stories"""
    if not results['structure_stats']['stories_by_phase']:
        return None

    # Get all expected phases
    expected_phases = [phase.value for phase in NarrativePhase]

    # Prepare data, ensuring all expected phases are included
    phase_names = []
    phase_percentages = []

    story_count = results['story_count']

    for phase in expected_phases:
        phase_names.append(phase)
        count = results['structure_stats']['stories_by_phase'].get(phase, 0)
        phase_percentages.append(count / story_count * 100)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(phase_names, phase_percentages, color='#27ae60', alpha=0.8, edgecolor='black')

    # Add value labels INSIDE the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
                 f'{height:.1f}%', ha='center', va='center',
                 fontsize=ANNOTATION_SIZE, color='white', fontweight='bold')

    plt.xlabel('Narrative Phase', fontsize=MEDIUM_SIZE)
    plt.ylabel('Percentage of Stories', fontsize=MEDIUM_SIZE)
    plt.ylim(0, 100)  # 100% maximum
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=20, fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'narrative_phase_distribution.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return save_path


# Function to extract bounding box data from story content
def extract_bounding_boxes(story_content):
    """Extract bounding box coordinates from the story tags"""
    # Pattern to match bounding box coordinates in image tags
    bb_pattern = r'<gdo\s+(?:char|obj)\d+\s+bb="([^"]+)"[^>]*>(.*?)</gdo>'

    # Extract all bounding boxes with their entity IDs
    matches = re.findall(bb_pattern, story_content, re.DOTALL)

    entity_boxes = []
    for bb_coords, entity_text in matches:
        # Parse coordinates (typically in format "x1,y1,x2,y2")
        try:
            coords = [float(c) for c in bb_coords.split(',')]
            if len(coords) == 4:
                entity_id = re.search(r'<gdo\s+(char|obj)(\d+)', entity_text)
                if entity_id:
                    entity_type, entity_num = entity_id.groups()
                    entity_id = f"{entity_type}{entity_num}"

                    # Calculate width and height
                    width = coords[2] - coords[0]
                    height = coords[3] - coords[1]

                    entity_boxes.append({
                        'entity_id': entity_id,
                        'bbox': coords,
                        'width': width,
                        'height': height,
                        'area': width * height,
                        'text': entity_text.strip()
                    })
        except:
            continue

    return entity_boxes

# Function to analyze bounding box consistency
def analyze_bounding_box_consistency(results):
    """Analyze consistency of bounding boxes for the same entities across frames"""
    entity_box_sizes = defaultdict(list)

    for story_id in results.get('story_ids', []):
        story_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_story.txt")

        if not os.path.exists(story_file):
            continue

        with open(story_file, 'r', encoding='utf-8') as f:
            story_content = f.read()

        bboxes = extract_bounding_boxes(story_content)

        for box in bboxes:
            entity_box_sizes[box['entity_id']].append(box['area'])

    # Calculate consistency metrics for entities with multiple appearances
    consistency_data = []
    for entity_id, areas in entity_box_sizes.items():
        if len(areas) > 1:
            # Calculate coefficient of variation (lower means more consistent)
            mean_area = statistics.mean(areas)
            if mean_area > 0:
                std_dev = statistics.stdev(areas) if len(areas) > 1 else 0
                cv = std_dev / mean_area

                consistency_data.append({
                    'entity_id': entity_id,
                    'cv': cv,
                    'mean_area': mean_area,
                    'std_dev': std_dev,
                    'appearances': len(areas)
                })

    return consistency_data

# Function to track grounding tags across narrative phases
def analyze_grounding_tags_by_phase(results):
    """Analyze distribution of grounding tags across narrative phases"""
    # Get all expected phases for consistent ordering
    expected_phases = [phase.value for phase in NarrativePhase]

    # Initialize storage for tag counts by phase
    phase_tag_counts = {
        phase: {'char': 0, 'obj': 0, 'action': 0, 'location': 0, 'total': 0, 'word_count': 0}
        for phase in expected_phases
    }

    # Process each story
    stories_processed = 0
    for story_id in results.get('story_ids', []):
        # Get story content
        story_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_story.txt")
        if not os.path.exists(story_file):
            continue

        with open(story_file, 'r', encoding='utf-8') as f:
            story_content = f.read()

        # Get CoT file for narrative phases
        cot_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_cot.txt")
        if not os.path.exists(cot_file):
            continue

        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_content = f.read()

        # Parse CoT and story
        cot = StoryReasoningUtil.parse_cot(cot_content)
        story = StoryReasoningUtil.parse_story(story_content)

        if not cot or not cot.narrative_structure or not cot.narrative_structure.phases or not story:
            continue

        stories_processed += 1

        # Create a mapping of image numbers to their text content
        image_texts = {img.image_number: img.text for img in story.images}

        # Map narrative phases to images
        phase_to_images = {}
        for phase_data in cot.narrative_structure.phases:
            if 'Narrative Phase' in phase_data and 'Images' in phase_data:
                phase_name = phase_data['Narrative Phase'].value
                image_nums = []

                # Extract image numbers (handling various formats)
                if isinstance(phase_data['Images'], list):
                    for img in phase_data['Images']:
                        try:
                            # Try direct integer conversion
                            image_nums.append(int(img))
                        except ValueError:
                            # Try extracting number from strings like "Image 1"
                            match = re.search(r'(\d+)', str(img))
                            if match:
                                image_nums.append(int(match.group(1)))

                elif isinstance(phase_data['Images'], str):
                    # Handle comma-separated format and extract numbers
                    img_strs = phase_data['Images'].split(',')
                    for img_str in img_strs:
                        match = re.search(r'(\d+)', img_str.strip())
                        if match:
                            image_nums.append(int(match.group(1)))

                phase_to_images[phase_name] = image_nums

        # Process grounding tags by narrative phase using the images it contains
        for phase, image_nums in phase_to_images.items():
            # Combine text from all images in this phase
            phase_text = ""
            for img_num in image_nums:
                if img_num in image_texts:
                    phase_text += image_texts[img_num] + " "

            if not phase_text:
                continue

            # Count char tags
            char_tags = re.findall(r'<gdo char\d+[^>]*>.*?</gdo>', phase_text, re.DOTALL)
            phase_tag_counts[phase]['char'] += len(char_tags)

            # Count obj tags
            obj_tags = re.findall(r'<gdo obj\d+[^>]*>.*?</gdo>', phase_text, re.DOTALL)
            phase_tag_counts[phase]['obj'] += len(obj_tags)

            # Count action tags
            action_tags = re.findall(r'<gda[^>]*>.*?</gda>', phase_text, re.DOTALL)
            phase_tag_counts[phase]['action'] += len(action_tags)

            # Count location tags
            location_tags = re.findall(r'<gdl[^>]*>.*?</gdl>', phase_text, re.DOTALL)
            phase_tag_counts[phase]['location'] += len(location_tags)

            # Count total tags
            phase_tag_counts[phase]['total'] += (len(char_tags) + len(obj_tags) +
                                                 len(action_tags) + len(location_tags))

            # Count words for density calculation
            clean_text = StoryReasoningUtil.strip_story_tags(phase_text)
            words = re.findall(r'\b\w+\b', clean_text)
            phase_tag_counts[phase]['word_count'] += len(words)

    # Calculate tag density (tags per 100 words)
    phase_tag_density = {}
    for phase, counts in phase_tag_counts.items():
        if counts['word_count'] > 0:
            phase_tag_density[phase] = {
                'char': (counts['char'] / counts['word_count']) * 100,
                'obj': (counts['obj'] / counts['word_count']) * 100,
                'action': (counts['action'] / counts['word_count']) * 100,
                'location': (counts['location'] / counts['word_count']) * 100,
                'total': (counts['total'] / counts['word_count']) * 100,
                'raw_counts': counts
            }
        else:
            phase_tag_density[phase] = {
                'char': 0, 'obj': 0, 'action': 0, 'location': 0, 'total': 0,
                'raw_counts': counts
            }

    return phase_tag_density

def create_grounding_density_chart(phase_tag_density, output_dir):
    """Create line graph showing density of grounding tags across narrative phases"""
    if not phase_tag_density:
        return None

    # Order phases in the standard narrative sequence
    ordered_phases = [p.value for p in NarrativePhase]
    phases = [p for p in ordered_phases if p in phase_tag_density]

    # Check if we have any data
    has_data = False
    for phase in phases:
        if (phase_tag_density[phase]['char'] > 0 or phase_tag_density[phase]['obj'] > 0 or
                phase_tag_density[phase]['action'] > 0 or phase_tag_density[phase]['location'] > 0):
            has_data = True
            break

    if not has_data:
        print("Warning: No grounding tag data found for any narrative phase")
        # Return None or create a message on the chart

    # Prepare data
    char_density = [phase_tag_density[p]['char'] for p in phases]
    obj_density = [phase_tag_density[p]['obj'] for p in phases]
    action_density = [phase_tag_density[p]['action'] for p in phases]
    location_density = [phase_tag_density[p]['location'] for p in phases]

    plt.figure(figsize=(12, 8))

    # Plot lines for each tag type
    plt.plot(phases, char_density, 'o-', color='#16a085', linewidth=2, markersize=8, label='Character References')
    plt.plot(phases, obj_density, 's-', color='#d35400', linewidth=2, markersize=8, label='Object References')
    plt.plot(phases, action_density, '^-', color='#c0392b', linewidth=2, markersize=8, label='Action References')
    plt.plot(phases, location_density, 'D-', color='#8e44ad', linewidth=2, markersize=8, label='Location References')

    plt.title('Grounding Tag Density Across Narrative Phases', fontsize=LARGE_SIZE)
    plt.xlabel('Narrative Phase', fontsize=MEDIUM_SIZE)
    plt.ylabel('Tags per 100 Words', fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=MEDIUM_SIZE)
    plt.xticks(rotation=20, fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)

    # Set a reasonable y-axis limit based on data
    max_density = max(
        max(char_density, default=0),
        max(obj_density, default=0),
        max(action_density, default=0),
        max(location_density, default=0)
    )

    if max_density > 0:
        plt.ylim(0, max_density * 1.2)  # Add 20% padding
    else:
        plt.ylim(0, 10)  # Default range if no data

    # Add value annotations
    for i, phase in enumerate(phases):
        # Character density
        if char_density[i] > 0:
            plt.annotate(f'{char_density[i]:.1f}',
                         (phases[i], char_density[i]),
                         textcoords="offset points",
                         xytext=(0,7),
                         ha='center',
                         fontsize=ANNOTATION_SIZE,
                         color='#16a085')

        # Action density
        if action_density[i] > 0:
            plt.annotate(f'{action_density[i]:.1f}',
                         (phases[i], action_density[i]),
                         textcoords="offset points",
                         xytext=(0,7),
                         ha='center',
                         fontsize=ANNOTATION_SIZE,
                         color='#c0392b')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'grounding_density_by_phase.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path


def analyze_entity_sentiment(results):
    """Analyze sentiment associated with characters across narrative phases"""
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()

    # Get all expected phases for consistent ordering
    expected_phases = [phase.value for phase in NarrativePhase]

    # Initialize storage for character sentiments by phase
    char_sentiment_by_phase = {
        phase: defaultdict(list) for phase in expected_phases
    }

    # Process each story
    for story_id in tqdm(results.get('story_ids', []), desc="Analyzing Sentiment"):
        # Get story content
        story_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_story.txt")
        if not os.path.exists(story_file):
            continue

        with open(story_file, 'r', encoding='utf-8') as f:
            story_content = f.read()

        # Get CoT file for narrative phases
        cot_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_cot.txt")
        if not os.path.exists(cot_file):
            continue

        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_content = f.read()

        # Parse CoT and story
        cot = StoryReasoningUtil.parse_cot(cot_content)
        story = StoryReasoningUtil.parse_story(story_content)

        if not cot or not cot.narrative_structure or not cot.narrative_structure.phases or not story:
            continue

        # Create a mapping of image numbers to their text content
        image_texts = {img.image_number: img.text for img in story.images}

        # Map narrative phases to images
        phase_to_images = {}
        for phase_data in cot.narrative_structure.phases:
            if 'Narrative Phase' in phase_data and 'Images' in phase_data:
                phase_name = phase_data['Narrative Phase'].value
                image_nums = []

                # Extract image numbers (handling various formats)
                if isinstance(phase_data['Images'], list):
                    for img in phase_data['Images']:
                        try:
                            # Try direct integer conversion
                            image_nums.append(int(img))
                        except ValueError:
                            # Try extracting number from strings like "Image 1"
                            match = re.search(r'(\d+)', str(img))
                            if match:
                                image_nums.append(int(match.group(1)))

                elif isinstance(phase_data['Images'], str):
                    # Handle comma-separated format and extract numbers
                    img_strs = phase_data['Images'].split(',')
                    for img_str in img_strs:
                        match = re.search(r'(\d+)', img_str.strip())
                        if match:
                            image_nums.append(int(match.group(1)))

                phase_to_images[phase_name] = image_nums

        # Process sentiment by narrative phase using the images it contains
        for phase, image_nums in phase_to_images.items():
            # Combine text from all images in this phase
            phase_text = ""
            for img_num in image_nums:
                if img_num in image_texts:
                    phase_text += image_texts[img_num] + " "

            if not phase_text:
                continue

            # Clean the text of tags for overall sentiment analysis
            clean_segment = StoryReasoningUtil.strip_story_tags(phase_text)

            # Calculate overall sentiment for the phase
            overall_sentiment = sia.polarity_scores(clean_segment)['compound']
            char_sentiment_by_phase[phase]['overall'].append(overall_sentiment)

            # Extract character tags to analyze character-specific sentiment
            char_emotions = defaultdict(list)

            # Use regex to find character references with context
            char_tags = re.finditer(r'<gdo char(\d+)[^>]*>(.*?)</gdo>', phase_text, re.DOTALL)
            for match in char_tags:
                char_id = match.group(1)

                # Get context around the character mention (up to 100 chars before and after)
                char_pos = match.start()
                context_start = max(0, char_pos - 100)
                context_end = min(len(phase_text), char_pos + 100)
                context = phase_text[context_start:context_end]

                # Clean the context text of any tags
                clean_context = StoryReasoningUtil.strip_story_tags(context)

                # Calculate sentiment for this context
                sentiment = sia.polarity_scores(clean_context)['compound']
                char_emotions[char_id].append(sentiment)

            # Calculate average sentiment for each character in this phase
            for char_id, sentiments in char_emotions.items():
                if sentiments:
                    avg_sentiment = statistics.mean(sentiments)
                    char_sentiment_by_phase[phase][char_id].append(avg_sentiment)

    # Average sentiments across stories
    avg_sentiment_by_phase = {}
    for phase in expected_phases:
        avg_sentiment_by_phase[phase] = {}

        # Calculate average sentiment for each character
        for char_id, sentiments in char_sentiment_by_phase[phase].items():
            if sentiments:
                avg_sentiment_by_phase[phase][char_id] = statistics.mean(sentiments)

        # Calculate overall average for the phase
        if 'overall' in char_sentiment_by_phase[phase] and char_sentiment_by_phase[phase]['overall']:
            avg_sentiment_by_phase[phase]['overall'] = statistics.mean(char_sentiment_by_phase[phase]['overall'])
        else:
            avg_sentiment_by_phase[phase]['overall'] = 0

    return avg_sentiment_by_phase

def create_entity_sentiment_chart(sentiment_data, output_dir):
    """Create chart showing character sentiment evolution across narrative phases with frequency information"""
    if not sentiment_data:
        return None

    # Order phases in the standard narrative sequence
    ordered_phases = [p.value for p in NarrativePhase]
    phases = [p for p in ordered_phases if p in sentiment_data]

    # Get overall sentiment for each phase
    overall_sentiment = [sentiment_data[p].get('overall', 0) for p in phases]

    # Get positive and negative sentiments for stacked area
    positive_sentiment = []
    negative_sentiment = []

    # Track frequency of positive vs negative contexts
    positive_frequency = []
    negative_frequency = []

    for phase in phases:
        phase_values = [s for s in sentiment_data[phase].values() if s != sentiment_data[phase].get('overall', 0)]
        pos_values = [v for v in phase_values if v > 0]
        neg_values = [v for v in phase_values if v < 0]

        # Calculate mean sentiment intensities
        positive_sentiment.append(statistics.mean(pos_values) if pos_values else 0)
        negative_sentiment.append(abs(statistics.mean(neg_values)) if neg_values else 0)

        # Calculate frequencies (proportion of positive vs negative)
        total_contexts = len(pos_values) + len(neg_values)
        if total_contexts > 0:
            positive_frequency.append(len(pos_values) / total_contexts)
            negative_frequency.append(len(neg_values) / total_contexts)
        else:
            positive_frequency.append(0)
            negative_frequency.append(0)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Top subplot - sentiment intensity
    ax1.fill_between(phases, 0, positive_sentiment, alpha=0.6, color='#2ecc71', label='Positive Sentiment')
    ax1.fill_between(phases, 0, [-v for v in negative_sentiment], alpha=0.6, color='#e74c3c', label='Negative Sentiment')
    ax1.plot(phases, overall_sentiment, 'o-', linewidth=2, markersize=8, color='#3498db', label='Overall Sentiment')

    ax1.set_title('Character Sentiment Evolution Across Narrative Phases', fontsize=LARGE_SIZE)
    ax1.set_ylabel('Sentiment Intensity', fontsize=MEDIUM_SIZE)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=MEDIUM_SIZE)
    ax1.set_xticks([])  # Remove x-ticks from top plot

    # Add annotations for overall sentiment
    for i, (phase, sentiment) in enumerate(zip(phases, overall_sentiment)):
        ax1.annotate(f'{sentiment:.2f}',
                     (phase, sentiment),
                     textcoords="offset points",
                     xytext=(0,7),
                     ha='center',
                     fontsize=ANNOTATION_SIZE,
                     color='#3498db')

    # Bottom subplot - sentiment frequency
    bar_width = 0.35
    x = np.arange(len(phases))

    # Create stacked bar chart for frequency
    ax2.bar(x, positive_frequency, bar_width, color='#2ecc71', alpha=0.8, label='Positive Contexts')
    ax2.bar(x, negative_frequency, bar_width, bottom=positive_frequency, color='#e74c3c', alpha=0.8,
            label='Negative Contexts')

    # Add percentage labels to the bars
    for i, (pos, neg) in enumerate(zip(positive_frequency, negative_frequency)):
        if pos > 0.05:  # Only show label if segment is large enough
            ax2.annotate(f'{pos:.0%}',
                         xy=(i, pos/2),
                         ha='center',
                         va='center',
                         color='white',
                         fontweight='bold',
                         fontsize=SMALL_SIZE)
        if neg > 0.05:  # Only show label if segment is large enough
            ax2.annotate(f'{neg:.0%}',
                         xy=(i, pos + neg/2),
                         ha='center',
                         va='center',
                         color='white',
                         fontweight='bold',
                         fontsize=SMALL_SIZE)

    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Context Frequency', fontsize=MEDIUM_SIZE)
    ax2.set_xlabel('Narrative Phase', fontsize=MEDIUM_SIZE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases, rotation=20, fontsize=SMALL_SIZE)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=SMALL_SIZE, loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'entity_sentiment_evolution.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path

def create_ungrounded_pronouns_chart(stats, output_dir):
    """Create bar chart showing ungrounded percentage for each pronoun"""
    # Extract pronoun data
    ungrounded_data = stats.get('ungrounded_by_pronoun', {}).get('value', {})

    if not ungrounded_data:
        return None

    # Sort pronouns by ungrounded percentage (descending)
    sorted_pronouns = sorted(ungrounded_data.items(), key=lambda x: x[1], reverse=True)
    pronouns = [p[0] for p in sorted_pronouns]
    percentages = [p[1] for p in sorted_pronouns]

    # Create color mapping based on pronoun type
    subject_pronouns = ['I', 'you', 'he', 'she', 'it', 'we', 'they']
    possessive_pronouns = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
    object_pronouns = ['me', 'you', 'him', 'her', 'it', 'us', 'them']

    colors = []
    for pronoun in pronouns:
        if pronoun.lower() in [p.lower() for p in subject_pronouns]:
            colors.append('#3498db')  # Blue for subject pronouns
        elif pronoun.lower() in [p.lower() for p in possessive_pronouns]:
            colors.append('#2ecc71')  # Green for possessive pronouns
        elif pronoun.lower() in [p.lower() for p in object_pronouns]:
            colors.append('#e74c3c')  # Red for object pronouns
        else:
            colors.append('#95a5a6')  # Gray for other cases

    plt.figure(figsize=(12, 8))
    bars = plt.bar(pronouns, percentages, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold')

    plt.title('Ungrounded Percentage by Pronoun', fontsize=LARGE_SIZE)
    plt.xlabel('Pronoun', fontsize=MEDIUM_SIZE)
    plt.ylabel('Ungrounded Percentage', fontsize=MEDIUM_SIZE)
    plt.ylim(0, 100 + 5)  # Add some space for labels
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)

    # Add legend for pronoun types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', alpha=0.8, label='Subject Pronouns'),
        Patch(facecolor='#2ecc71', edgecolor='black', alpha=0.8, label='Possessive Pronouns'),
        Patch(facecolor='#e74c3c', edgecolor='black', alpha=0.8, label='Object Pronouns')
    ]
    plt.legend(handles=legend_elements, fontsize=MEDIUM_SIZE)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'ungrounded_pronouns.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path

# Function to create comparative charts for subject vs. possessive pronouns
def create_subject_vs_possessive_chart(stats, output_dir):
    """Create comparative charts for subject vs. possessive pronoun groundedness"""
    # Extract data
    subject_data = stats.get('ungrounded_by_subject_pronoun', {}).get('value', {})
    possessive_data = stats.get('ungrounded_by_possessive_pronoun', {}).get('value', {})

    if not subject_data or not possessive_data:
        return None

    # Create a 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # PANEL 1: Subject Pronouns
    sorted_subject = sorted(subject_data.items(), key=lambda x: x[1], reverse=True)
    subject_pronouns = [p[0] for p in sorted_subject]
    subject_percentages = [p[1] for p in sorted_subject]

    bars1 = ax1.bar(subject_pronouns, subject_percentages, color='#3498db', alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold')

    ax1.set_title('Subject Pronouns', fontsize=LARGE_SIZE)
    ax1.set_xlabel('Pronoun', fontsize=MEDIUM_SIZE)
    ax1.set_ylabel('Ungrounded Percentage', fontsize=MEDIUM_SIZE)
    ax1.set_ylim(0, 100 + 5)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)

    # PANEL 2: Possessive Pronouns
    sorted_possessive = sorted(possessive_data.items(), key=lambda x: x[1], reverse=True)
    possessive_pronouns = [p[0] for p in sorted_possessive]
    possessive_percentages = [p[1] for p in sorted_possessive]

    bars2 = ax2.bar(possessive_pronouns, possessive_percentages, color='#2ecc71', alpha=0.8, edgecolor='black')

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold')

    ax2.set_title('Possessive Pronouns', fontsize=LARGE_SIZE)
    ax2.set_xlabel('Pronoun', fontsize=MEDIUM_SIZE)
    ax2.set_ylabel('Ungrounded Percentage', fontsize=MEDIUM_SIZE)
    ax2.set_ylim(0, 100 + 5)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)

    plt.suptitle('Comparison of Ungrounded Subject vs. Possessive Pronouns', fontsize=LARGE_SIZE+2)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle

    save_path = os.path.join(output_dir, 'subject_vs_possessive_pronouns.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path


def create_pronoun_person_comparison_chart(stats, output_dir):
    """Create improved chart comparing third-person vs. first/second person groundedness"""
    # Extract data
    pronoun_data = stats.get('ungrounded_by_pronoun', {}).get('value', {})

    if not pronoun_data:
        return None

    # Categorize pronouns
    first_person = {'I', 'me', 'my', 'we', 'us', 'our'}
    second_person = {'you', 'your'}
    third_person = {'he', 'she', 'it', 'they', 'him', 'her', 'its', 'their', 'them'}

    # Group the data
    categories = {
        'First Person': [pronoun_data.get(p, 0) for p in first_person if p in pronoun_data],
        'Second Person': [pronoun_data.get(p, 0) for p in second_person if p in pronoun_data],
        'Third Person': [pronoun_data.get(p, 0) for p in third_person if p in pronoun_data]
    }

    # Calculate averages
    avg_by_category = {}
    for category, values in categories.items():
        if values:
            avg_by_category[category] = statistics.mean(values)
        else:
            avg_by_category[category] = 0

    # Create figure with subplot layout - top part for chart, bottom for legend
    fig = plt.figure(figsize=(12, 9))

    # Create the main bar chart (using 80% of the vertical space)
    ax1 = fig.add_axes([0.1, 0.3, 0.8, 0.6])  # [left, bottom, width, height]

    # Prepare data for plotting
    categories = list(avg_by_category.keys())
    averages = [avg_by_category[c] for c in categories]

    # Colors
    colors = ['#e74c3c', '#f39c12', '#3498db']  # Red, Orange, Blue

    # Create the bars
    bars = ax1.bar(categories, averages, color=colors, alpha=0.8, edgecolor='black', width=0.6)

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', fontsize=ANNOTATION_SIZE, fontweight='bold')

    # Set chart title and labels
    ax1.set_title('Average Ungrounded Percentage by Pronoun Person', fontsize=LARGE_SIZE)
    ax1.set_ylabel('Average Ungrounded Percentage', fontsize=MEDIUM_SIZE)
    ax1.set_ylim(0, 105)  # Set y-axis limit with space for labels
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_axisbelow(True)  # Put grid lines behind bars

    # Format x-axis
    ax1.tick_params(axis='both', which='major', labelsize=MEDIUM_SIZE)

    # Create a separate text area (using 20% of space at bottom) for the pronoun lists
    ax2 = fig.add_axes([0.1, 0.05, 0.8, 0.15])  # [left, bottom, width, height]
    ax2.axis('off')  # Hide axes for text area

    # Create formatted text explanations
    first_person_text = "First Person: " + ", ".join(sorted(first_person))
    second_person_text = "Second Person: " + ", ".join(sorted(second_person))
    third_person_text = "Third Person: " + ", ".join(sorted(third_person))

    # Add text boxes with matching colors
    ax2.text(0.5, 1, first_person_text, ha='center', va='center', fontsize=MEDIUM_SIZE,
             bbox=dict(facecolor='#e74c3c', alpha=0.2, edgecolor='#e74c3c', pad=6))

    ax2.text(0.5, 0.6, second_person_text, ha='center', va='center', fontsize=MEDIUM_SIZE,
             bbox=dict(facecolor='#f39c12', alpha=0.2, edgecolor='#f39c12', pad=6))

    ax2.text(0.5, 0.2, third_person_text, ha='center', va='center', fontsize=MEDIUM_SIZE,
             bbox=dict(facecolor='#3498db', alpha=0.2, edgecolor='#3498db', pad=6))

    # Save the chart
    save_path = os.path.join(output_dir, 'pronoun_person_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path


# Function to analyze cross-frame reasoning complexity
def analyze_cross_frame_reasoning(results):
    """Analyze logical connections between frames based on entity persistence"""
    # Initialize storage for frame connections
    frame_connections = defaultdict(lambda: {'persistent': 0, 'new': 0, 'total': 0, 'count': 0})

    # Process each story
    for story_id in results.get('story_ids', []):
        # Get CoT file for frame analysis
        cot_file = os.path.join(results.get('dataset_path', ''), f"{story_id}_cot.txt")
        if not os.path.exists(cot_file):
            continue

        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_content = f.read()

        # Parse CoT
        cot = StoryReasoningUtil.parse_cot(cot_content)
        if not cot or not cot.images or len(cot.images) < 2:
            continue

        # Track entities across frames
        entity_frames = defaultdict(set)

        # Collect all entities by frame
        for frame_idx, image_analysis in enumerate(cot.images):
            # Characters in this frame
            for char in image_analysis.characters:
                entity_frames[f"char{char.character_id}"].add(frame_idx)

            # Objects in this frame
            for obj in image_analysis.objects:
                # Skip background objects
                if not obj.object_id.startswith('bg'):
                    entity_frames[f"obj{obj.object_id}"].add(frame_idx)

        # For each consecutive frame pair, calculate persistent vs. new entities
        for i in range(1, len(cot.images)):
            current_frame = i
            prev_frame = i - 1

            # Count entities in previous frame
            prev_entities = set()
            for entity, frames in entity_frames.items():
                if prev_frame in frames:
                    prev_entities.add(entity)

            # Count entities in current frame
            current_entities = set()
            for entity, frames in entity_frames.items():
                if current_frame in frames:
                    current_entities.add(entity)

            # Calculate persistent and new entities
            persistent = len(prev_entities.intersection(current_entities))
            new_entities = len(current_entities - prev_entities)
            total = len(current_entities)

            # Store results
            frame_connections[i]['persistent'] += persistent
            frame_connections[i]['new'] += new_entities
            frame_connections[i]['total'] += total
            frame_connections[i]['count'] += 1  # Track number of stories with this transition

    # Calculate averages
    avg_frame_connections = {}

    for frame_idx, counts in frame_connections.items():
        # Use story count per transition rather than global story count
        story_count = counts['count'] if counts['count'] > 0 else 1

        avg_frame_connections[frame_idx] = {
            'avg_persistent': counts['persistent'] / story_count,
            'avg_new': counts['new'] / story_count,
            'avg_total': counts['total'] / story_count,
            'raw_counts': counts
        }

    return avg_frame_connections

def create_cross_frame_reasoning_chart(frame_connections, output_dir):
    """Create bar chart showing logical connections between frames"""
    if not frame_connections:
        return None

    # Get frame indices (excluding frame 0 which is the first frame)
    frames = sorted(frame_connections.keys())

    # Prepare data
    persistent_entities = [frame_connections[f]['avg_persistent'] for f in frames]
    new_entities = [frame_connections[f]['avg_new'] for f in frames]

    plt.figure(figsize=(12, 8))

    # Set up bar positions
    bar_width = 0.35
    frame_positions = list(range(len(frames)))

    # Create grouped bars
    bars1 = plt.bar([p - bar_width/2 for p in frame_positions], persistent_entities, bar_width,
                    label='Persistent Entities', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = plt.bar([p + bar_width/2 for p in frame_positions], new_entities, bar_width,
                    label='New Entities', color='#e74c3c', alpha=0.8, edgecolor='black')

    plt.title('Cross-Frame Reasoning Complexity', fontsize=LARGE_SIZE)
    plt.xlabel('Frame Transition', fontsize=MEDIUM_SIZE)
    plt.ylabel('Average Number of Entities', fontsize=MEDIUM_SIZE)
    plt.xticks([p for p in frame_positions], [f'Frame {i-1} → {i}' for i in frames],
               fontsize=SMALL_SIZE, rotation=45)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.legend(fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value annotations
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero values
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.1f}', ha='center', fontsize=ANNOTATION_SIZE)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        if height > 0:  # Only add labels for non-zero values
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                     f'{height:.1f}', ha='center', fontsize=ANNOTATION_SIZE)

    # Set a reasonable y-axis limit based on data
    max_height = max(max(persistent_entities, default=0), max(new_entities, default=0))
    plt.ylim(0, max_height * 1.2)  # Add 20% padding

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_frame_reasoning.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path

# Chart creation functions
def create_bbox_consistency_chart(consistency_data, output_dir):
    """Create violin plot showing bounding box size consistency for entities"""
    if not consistency_data:
        return None

    plt.figure(figsize=(12, 8))

    # Extract CV values and entity types
    cv_values = [d['cv'] for d in consistency_data]
    entity_types = [d['entity_id'].startswith('char') for d in consistency_data]

    # Create DataFrame for seaborn
    df = pd.DataFrame({
        'Coefficient of Variation': cv_values,
        'Entity Type': ['Character' if is_char else 'Object' for is_char in entity_types],
        'Appearances': [d['appearances'] for d in consistency_data]
    })

    # Create violin plot
    sns.violinplot(x='Entity Type', y='Coefficient of Variation', data=df,
                   inner='quartile', palette=['#9b59b6', '#f1c40f'])

    # Add jittered points for individual entities
    sns.stripplot(x='Entity Type', y='Coefficient of Variation', data=df,
                  size=4, color='.3', alpha=0.6)

    plt.title('Bounding Box Size Consistency Across Frames', fontsize=LARGE_SIZE)
    plt.xlabel('Entity Type', fontsize=MEDIUM_SIZE)
    plt.ylabel('Coefficient of Variation (lower = more consistent)', fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=SMALL_SIZE)
    plt.yticks(fontsize=SMALL_SIZE)

    # Add annotation explaining the metric
    plt.figtext(0.5, 0.01,
                'Lower coefficient of variation indicates more consistent entity size across frames.',
                ha='center', fontsize=SMALL_SIZE, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_path = os.path.join(output_dir, 'bbox_consistency.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path


def create_cross_frame_reasoning_chart(frame_connections, output_dir):
    """Create bar chart showing logical connections between frames"""
    if not frame_connections:
        return None

    # Get frame indices (excluding frame 0 which is the first frame)
    frames = sorted(frame_connections.keys())

    # Prepare data
    persistent_entities = [frame_connections[f]['avg_persistent'] for f in frames]
    new_entities = [frame_connections[f]['avg_new'] for f in frames]

    plt.figure(figsize=(12, 8))

    # Set up bar positions
    bar_width = 0.35
    frame_positions = list(range(len(frames)))

    # Create grouped bars
    bars1 = plt.bar([p - bar_width/2 for p in frame_positions], persistent_entities, bar_width,
                    label='Persistent Entities', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = plt.bar([p + bar_width/2 for p in frame_positions], new_entities, bar_width,
                    label='New Entities', color='#e74c3c', alpha=0.8, edgecolor='black')

    plt.title('Cross-Frame Reasoning Complexity', fontsize=LARGE_SIZE)
    plt.xlabel('Frame Transition', fontsize=MEDIUM_SIZE)
    plt.ylabel('Average Number of Entities', fontsize=MEDIUM_SIZE)
    plt.xticks([p for p in frame_positions], [f'Frame {i-1} → {i}' for i in frames], fontsize=SMALL_SIZE, rotation=20)
    plt.yticks(fontsize=SMALL_SIZE)
    plt.legend(fontsize=MEDIUM_SIZE)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value annotations
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', fontsize=ANNOTATION_SIZE)

    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.1f}', ha='center', fontsize=ANNOTATION_SIZE)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cross_frame_reasoning.pdf')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    return save_path


def create_visualizations(visualize_data, args):
    """Create and save all visualization charts for the dataset"""
    output_dir = args.output_dir
    dataset_dir = args.dataset_path
    os.makedirs(output_dir, exist_ok=True)
    results = visualize_data['raw_results']
    stats = visualize_data['stats']

    # Store dataset path for analysis functions that need it
    results['dataset_path'] = dataset_dir

    # If story_ids not populated, extract from results
    if not hasattr(results, 'story_ids') or not results.get('story_ids'):
        results['story_ids'] = []
        # Find all stories that have been processed
        for story_file in glob.glob(os.path.join(results['dataset_path'], "*_story.txt")):
            story_id = os.path.basename(story_file).split("_")[0]
            results['story_ids'].append(story_id)

    # Set up global plotting style
    setup_matplotlib_style()

    # Create each chart independently
    charts = {}

    # Original charts
    print("Creating frame distribution chart...")
    charts['frames_distribution'] = create_frames_distribution_chart(results, output_dir)

    print("Creating cross-frame consistency chart...")
    charts['cross_frame_consistency'] = create_cross_frame_consistency_chart(results, output_dir)

    print("Creating grounding stats chart...")
    charts['grounding_stats'] = create_grounding_stats_chart(stats, output_dir)

    print("Creating ungrounded subject pronouns chart...")
    charts['ungrounded_subject_pronouns'] = create_ungrounded_subject_pronouns_chart(results, output_dir)

    print("Creating ungrounded possessive pronouns chart...")
    charts['ungrounded_possessive_pronouns'] = create_ungrounded_possessive_pronouns_chart(results, output_dir)

    print("Creating ungrounded nouns chart...")
    charts['ungrounded_nouns'] = create_ungrounded_nouns_chart(results, output_dir)

    print("Creating entities by story length chart...")
    charts['entities_by_story_length'] = create_entities_by_story_length_chart(results, output_dir)

    print("Creating narrative phase distribution chart...")
    charts['narrative_phase_distribution'] = create_narrative_phase_chart(results, output_dir)

    print("Analyzing bounding box consistency...")
    consistency_data = analyze_bounding_box_consistency(results)
    charts['bbox_consistency'] = create_bbox_consistency_chart(consistency_data, output_dir)

    print("Analyzing grounding tags by narrative phase...")
    phase_tag_density = analyze_grounding_tags_by_phase(results)
    charts['grounding_density'] = create_grounding_density_chart(phase_tag_density, output_dir)

    print("Analyzing entity sentiment across narrative phases...")
    sentiment_data = analyze_entity_sentiment(results)
    charts['entity_sentiment'] = create_entity_sentiment_chart(sentiment_data, output_dir)

    print("Analyzing cross-frame reasoning complexity...")
    frame_connections = analyze_cross_frame_reasoning(results)
    charts['cross_frame_reasoning'] = create_cross_frame_reasoning_chart(frame_connections, output_dir)

    # Added charts for pronoun data
    print("Creating ungrounded pronouns chart...")
    charts['ungrounded_pronouns'] = create_ungrounded_pronouns_chart(stats, output_dir)

    print("Creating subject vs possessive pronouns chart...")
    charts['subject_vs_possessive'] = create_subject_vs_possessive_chart(stats, output_dir)

    print("Creating pronoun person comparison chart...")
    charts['pronoun_person_comparison'] = create_pronoun_person_comparison_chart(stats, output_dir)

    return output_dir


def export_metrics(stats, output_dir):
    """Export metrics to JSON file"""
    # Remove entity_distribution and narrative_phases metrics
    metrics_to_exclude = ['avg_unique_characters', 'avg_unique_objects', 'avg_unique_backgrounds',
                          'avg_unique_landmarks', 'avg_ungrounded_pronouns_by_frame',
                          'avg_ungrounded_nouns_by_frame']

    metrics_dict = {metric: {
        'value': stats[metric]['value'],
        'description': stats[metric]['description']
    } for metric in stats if metric not in metrics_to_exclude}

    # Creates the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    metrics_json_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics_json_path


def main():
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze Story Reasoning dataset and generate statistics')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='visualization_charts',
                        help='Directory for saving visualization charts')
    args = parser.parse_args()

    # Calculate statistics
    print("Analyzing dataset...")
    stats, visualize_data = analyze_dataset(args.dataset_path)

    # Export metrics to JSON
    print("Exporting metrics...")
    metrics_path = export_metrics(stats, args.output_dir)
    print(f"Metrics saved to {metrics_path}")

    # Create visualizations
    print("Creating visualization charts...")
    charts_dir = create_visualizations(visualize_data, args)
    print(f"Visualization charts saved to {charts_dir}")


if __name__ == "__main__":
    main()