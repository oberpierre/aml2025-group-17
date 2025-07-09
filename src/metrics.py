from collections import defaultdict
from utils import ENTITY_MAP
from typing import List, Tuple

class Metrics:
    """Simple class to track metrics during streaming."""
    def __init__(self):
        self.true_positives = defaultdict(int)
        self.true_negatives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.corrected_predictions = 0 # Predictions that were corrected in a subsequent NER execution
        self.time_to_first_detection = 0
        self.invocation_times_count = 0

    """Evaluates metrics for a sentence or document, given by its true BIO tags and 
    predicted BIO tags at specific times.

    Args:
        true_bio (List[str]): True BIO tags for the sentence or document.
        pred_bio (List[List[str]]): List of predicted BIO tags for the sentence or document at each NER execution time in order of execution.
            I.e. len(pred_bio) is the number of NER execution times and len(pred_bio[i]) is the number of tokens in the sentence or document at execution time.

    """
    def evaluate_metrics(self, ground_truth: Tuple[List[str], List[str]], pred_bio_runs: List[List[str]]):
        """Evaluate metrics for a sentence or document, given by its true BIO tags and predicted BIO tags at specific times."""
        tokens, true_bio = ground_truth
        self.invocation_times_count += len(pred_bio_runs)

        # Go through latest prediction run and count true positives, true negatives, false positives, and false negatives
        latest_predictions = pred_bio_runs[-1]
        for i, true_label in enumerate(true_bio):
            if latest_predictions[i] == true_label:
                if true_label == 'O':
                    # y and y_hat are 'O'
                    self.true_negatives[true_label] += 1
                else:
                    # y and y_hat are same NER
                    self.true_positives[true_label] += 1
            else:
                if latest_predictions[i] == 'O': # false negative
                    # y is a NER and y_hat is 'O'
                    self.false_negatives[true_label] += 1
                else: # false positive
                    # y is a 'O' or NER X and y_hat is a NER Y
                    self.false_positives[true_label] += 1

        # If the latest prediction run is shorter than the true BIO tags, we have false negatives unless there are no further entities
        if len(true_bio) > len(latest_predictions):
            for i in range(len(latest_predictions), len(true_bio)):
                if true_bio[i] != 'O':
                    self.false_negatives[true_bio[i]] += 1

        # TODO evaluate corrected predictions
        # TODO evaluate time to first detection

    def print_metrics(self):
        print("Metrics:")
        print(f"Total NER invocations: {self.invocation_times_count}")
        # Print table with entity types and their counts
        print(f"{'Entity Type':<20} {'TP':<10} {'TN':<10} {'FP':<10} {'FN':<10}")
        print("-" * 60)
        for entity_type in list(ENTITY_MAP.values()):
            tp = self.true_positives[entity_type]
            tn = self.true_negatives[entity_type]
            fp = self.false_positives[entity_type]
            fn = self.false_negatives[entity_type]
            
            if entity_type == 'O':
                # if true entity is 'O' (negative) -> there cannot be true positives or false negatives
                tp = self._format_highlight(str(tp)) if tp > 0 else "N/A"
                fn = self._format_highlight(str(fn)) if fn > 0 else "N/A"
            else:
                # if true entity is not 'O' (positive) -> there cannot be true negatives
                tn = self._format_highlight(str(tn)) if tn > 0 else "N/A"
            
            print(f"{entity_type:<20} {tp:<10} {tn:<10} {fp:<10} {fn:<10}")
        print(f"{'Total':<20} {sum(self.true_positives.values()):<10} {sum(self.true_negatives.values()):<10} {sum(self.false_positives.values()):<10} {sum(self.false_negatives.values()):<10}")
        print("-" * 60)

    @staticmethod
    def _format_highlight(text: str) -> str:
        """Highlight the text in a way that is suitable for printing."""
        return f"*{text}*?!"
