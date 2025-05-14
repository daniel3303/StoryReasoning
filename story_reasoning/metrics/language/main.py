from story_reasoning.metrics.language.bleu import Bleu, BleuType
from story_reasoning.metrics.language.cider import Cider
from story_reasoning.metrics.language.meteor import Meteor
from story_reasoning.metrics.language.rouge import Rouge


def generate_sample(number: int):
    """Generate a sample with variations based on the ID."""
    base_samples = [
        (
            "The dog chases a ball in the park.",
            [
                "The dog is playing with a ball in the park.",
                "A brown dog is running after a red ball."
            ]
        ),
        (
            "A cat sleeps on the windowsill in the afternoon sun.",
            [
                "The cat is resting by the window, enjoying the sunlight.",
                "An orange cat naps in the warm afternoon sunshine."
            ]
        ),
        (
            "Children play on the swings and slides at the playground.",
            [
                "Kids are having fun on the playground equipment.",
                "Several children enjoy the swings and slides."
            ]
        ),
        (
            "The bird sings a beautiful melody from the maple tree.",
            [
                "A small bird chirps melodiously from the branches.",
                "The bird is making sweet music in the maple tree."
            ]
        ),
        (
            "A red car drives down the busy street during rush hour.",
            [
                "The vehicle moves along the crowded road.",
                "A red sedan travels down the busy avenue."
            ]
        ),
        (
            "The chef prepares a delicious pasta dish in the kitchen.",
            [
                "A skilled cook makes pasta with fresh ingredients.",
                "The chef creates an Italian meal in the restaurant."
            ]
        ),
        (
            "Students study mathematics in the university library.",
            [
                "College students work on math problems in the library.",
                "Young scholars learn calculus in the quiet study area."
            ]
        ),
        (
            "The artist paints a landscape of mountains and lakes.",
            [
                "A painter creates a scenic view of nature.",
                "The artist captures mountains reflected in calm waters."
            ]
        ),
        (
            "The farmer harvests wheat in the golden field.",
            [
                "A farmer gathers the ripe wheat crop.",
                "The agricultural worker collects grain at harvest time."
            ]
        ),
        (
            "Tourists take photos of the ancient castle ruins.",
            [
                "Visitors photograph the historic castle remains.",
                "People capture images of the medieval fortress."
            ]
        ),
        (
            "The musician performs a classical piece on the piano.",
            [
                "A pianist plays a beautiful classical composition.",
                "The performer creates music on the grand piano."
            ]
        ),
        (
            "Scientists conduct experiments in the research laboratory.",
            [
                "Researchers perform tests in the modern lab.",
                "The science team carries out experiments with precision."
            ]
        ),
        (
            "A surfer rides the ocean waves at sunrise.",
            [
                "The surfer catches waves in the early morning.",
                "An experienced surfer glides through the morning surf."
            ]
        ),
        (
            "The gardener plants roses in the flower bed.",
            [
                "A person tends to the rose garden with care.",
                "The landscaper adds new roses to the garden."
            ]
        ),
        (
            "The baker creates fresh bread in the morning.",
            [
                "Fresh loaves of bread are prepared by the baker.",
                "The bakery produces warm bread at dawn."
            ]
        )
    ]

    # Use modulo to cycle through base samples
    base_idx = number % len(base_samples)
    return base_samples[base_idx]


def example_usage():
    """Example demonstrating how to use the metric wrappers with multiple samples."""
    # Initialize metrics
    metrics = {
        'BLEU-1': Bleu(bleu_type=BleuType.BLEU1, strip_grounding_tags=False),
        'BLEU-2': Bleu(bleu_type=BleuType.BLEU2, strip_grounding_tags=False),
        'BLEU-3': Bleu(bleu_type=BleuType.BLEU3, strip_grounding_tags=False),
        'BLEU-4': Bleu(bleu_type=BleuType.BLEU4, strip_grounding_tags=False),
        'METEOR': Meteor(strip_grounding_tags=False),
        'CIDEr': Cider(strip_grounding_tags=False),
        # 'SPICE': Spice(strip_grounding_tags=False),
        'ROUGE-L': Rouge(strip_grounding_tags=False),
    }

    # Generate samples
    num_samples = 15
    all_candidates = {}
    all_references = {}

    # Merge individual samples into a batch
    for sample_id in range(num_samples):
        candidate, references = generate_sample(sample_id)
        all_candidates[str(sample_id)] = candidate
        all_references[str(sample_id)] = references

    # Compute batch scores
    batch_scores = {}
    for name, metric in metrics.items():
        batch_scores[name] = metric.evaluate(all_references, all_candidates)

    return batch_scores


if __name__ == "__main__":
    scores = example_usage()
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")