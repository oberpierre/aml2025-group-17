from typing import List, Tuple

class Metrics:
    """Simple class to track metrics during streaming."""
    def __init__(self):
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
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
                if latest_predictions[i] != 'O':
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if latest_predictions[i] != 'O':
                    print(f"False positive at index {i}: predicted {latest_predictions[i]}, true {true_label}, for token '{tokens[i]}' ({tokens})")
                    self.false_positives += 1
                else:
                    print(f"False negative at index {i}: predicted {latest_predictions[i]}, true {true_label}, for token '{tokens[i]}' ({tokens})")
                    self.false_negatives += 1
        # If the latest prediction run is shorter than the true BIO tags, we have false negatives unless there are no further entities
        if len(true_bio) > len(latest_predictions):
            for i in range(len(latest_predictions), len(true_bio)):
                if true_bio[i] != 'O':
                    self.false_negatives += 1

        # TODO evaluate corrected predictions
        # TODO evaluate time to first detection

    def print_metrics(self):
        print(f"True Positives: {self.true_positives}")
        print(f"True Negatives: {self.true_negatives}")
        print(f"False Positives: {self.false_positives}")
        print(f"False Negatives: {self.false_negatives}")