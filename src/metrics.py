from collections import defaultdict
from utils import ENTITY_MAP
from typing import List, Tuple, Set
import pickle
import os

class Metrics:
    """Simple class to track metrics during streaming."""
    def __init__(self, known_entities: List[str] = ["PERSON", "ORG", "LOC"], entity_bucket_name: str = 'MISC'):
        self.true_positives = defaultdict(int)
        self.true_negatives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.classified_entities = []
        for entity_type in known_entities:
            self.classified_entities.append(f"B-{entity_type}")
            self.classified_entities.append(f"I-{entity_type}")
        self.catch_all_entities = [f"B-{entity_bucket_name}", f"I-{entity_bucket_name}"]
        self.false_positives_bucket = (defaultdict(int), defaultdict(int))  # (B-MISC, I-MISC)
        self.false_negatives = defaultdict(int)
        self.corrected_predictions = 0 # Predictions that were corrected in a subsequent NER execution
        self.time_to_first_detection = []
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
        # Copy the latest predictions to avoid modifying the original list for our hacky PERSON rectification
        latest_predictions = pred_bio_runs[-1].copy()
        for i, true_label in enumerate(true_bio):
            # NER model used classifies B-/I-PERSON as B-/I-PER, so we need to handle this case
            if latest_predictions[i].endswith('PER'):
                latest_predictions[i] = latest_predictions[i].replace('PER', 'PERSON')
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
                    if true_label == 'O':
                        # y is 'O' and y_hat is a NER
                        self.false_positives[true_label] += 1
                    else:
                        # y is NER X and y_hat is NER Y
                        self.false_positives[true_label] += 1
                        if true_label not in self.classified_entities and latest_predictions[i] in self.catch_all_entities:
                            index = 0 if latest_predictions[i].startswith('B-') else 1
                            # If the false positive is a catch-all entity, we count it separately
                            self.false_positives_bucket[index][true_label] += 1


        # If the latest prediction run is shorter than the true BIO tags, we have false negatives unless there are no further entities
        if len(true_bio) > len(latest_predictions):
            for i in range(len(latest_predictions), len(true_bio)):
                if true_bio[i] != 'O':
                    self.false_negatives[true_bio[i]] += 1

        # TODO evaluate corrected predictions

        # Evaluating time to first detection
        true_entities: Set[Tuple[int, int, str]] = set()
        i = 0
        while i < len(true_bio):
            if true_bio[i].startswith('B-'):
                start_idx = i
                entity_type = true_bio[i]
                i += 1
                # Find the end of this entity
                while i < len(true_bio) and true_bio[i] == f"I-{entity_type[2:]}":
                    i += 1
                end_idx = i - 1
                true_entities.add((start_idx, end_idx, entity_type))
            else:
                i += 1

        for pred_bio in pred_bio_runs:
            # len(pred_bio_runs) is the number of NER execution times and len(pred_bio_runs[i])
            # is the number of tokens in the sentence or document at execution time.
            detected_entities: Set[Tuple[int, int, str]] = set()
            actual_detection_time = len(pred_bio) - 1

            # Check if any entities are correctly detected for the first time in this run
            for start_idx, end_idx, entity_type in true_entities:
                entity_key = (start_idx, end_idx, entity_type)
                
                # Skip if prediction is too short
                if end_idx > actual_detection_time:
                    continue
                
                # Check if the entire entity is correctly predicted
                entity_correctly_predicted = True
                for pos in range(start_idx, end_idx + 1):
                    if pos == start_idx:
                        if entity_type in self.classified_entities:
                            if entity_type.endswith('PERSON'):
                                # Handle the PERSON rectification case
                                expected_tag = [f"B-PER"]
                            else:
                                expected_tag = [entity_type]
                        else:
                            expected_tag = [entity_type, self.catch_all_entities[0]]
                    else:
                        if entity_type in self.classified_entities:
                            if entity_type.endswith('PERSON'):
                                expected_tag = [f"I-PER"]
                            else:
                                expected_tag = [f"I-{entity_type[2:]}"]
                        else:
                            expected_tag = [f"I-{entity_type[2:]}", self.catch_all_entities[1]]
                    
                    if pred_bio[pos] not in expected_tag:
                        entity_correctly_predicted = False
                        break
                
                if entity_correctly_predicted:
                    detected_entities.add(entity_key)
                    time_diff = actual_detection_time - end_idx
                    self.time_to_first_detection.append(time_diff)
        
            # Removing already detected entities from true_entities to not process them again
            true_entities -= detected_entities
    
    def _calculate_fpr(self) -> float:
        """Calculate False Positive Rate (FPR) over all entity types."""
        total_fp = sum(self.false_positives.values())
        total_tn = sum(self.true_negatives.values())
        if total_fp + total_tn == 0:
            return 0.0
        return total_fp / (total_fp + total_tn)

    def _calculate_fnr(self) -> float:
        """Calculate False Negative Rate (FNR) over all entity types."""
        total_fn = sum(self.false_negatives.values())
        total_tp = sum(self.true_positives.values())
        if total_fn + total_tp == 0:
            return 0.0
        return total_fn / (total_fn + total_tp)

    def print_metrics(self):
        print("Metrics:")
        print(f"Total NER invocations: {self.invocation_times_count}")
        avg_ttfd = f"{sum(self.time_to_first_detection) / len(self.time_to_first_detection):.2f}" if self.time_to_first_detection else 'N/A'
        print(f"Avg TTFD: {avg_ttfd}")
        print(f"FPR@FNR: {self._calculate_fpr():.4f}@{self._calculate_fnr():.4f}")
        # Print table with entity types and their counts
        print(f"{'Entity Type':<20} {'TP':<10} {'TN':<10} {'FP (#B-/I-MISC)':<20} {'FN':<10}")
        print("-" * 70)
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

            # Displaying false positives for catch-all entities separately in format: "{false_positive} ({B-catch_all_entity}/I-catch_all_entity})"
            bucket_entities = self.false_positives_bucket[0][entity_type], self.false_positives_bucket[1][entity_type]
            fp_display = f"{fp} ({bucket_entities[0]}/{bucket_entities[1]})" if bucket_entities[0] + bucket_entities[1] > 0 else str(fp)

            print(f"{entity_type:<20} {tp:<10} {tn:<10} {fp_display:<20} {fn:<10}")

        total_fp_display = f"{sum(self.false_positives.values())} ({sum(self.false_positives_bucket[0].values())}/{sum(self.false_positives_bucket[1].values())})"
        print(f"{'Total':<20} {sum(self.true_positives.values()):<10} {sum(self.true_negatives.values()):<10} {total_fp_display:<20} {sum(self.false_negatives.values()):<10}")
        print("-" * 70)

    @staticmethod
    def _format_highlight(text: str) -> str:
        """Highlight the text in a way that is suitable for printing."""
        return f"*{text}*?!"
    
    def save_metrics(self, filename: str):
        """Save all metrics data to a file."""
        metrics_data = {
            'true_positives': dict(self.true_positives),
            'true_negatives': dict(self.true_negatives),
            'false_positives': dict(self.false_positives),
            'catch_all_entities': self.catch_all_entities,
            'false_positives_bucket': (dict(self.false_positives_bucket[0]), dict(self.false_positives_bucket[1])),
            'false_negatives': dict(self.false_negatives),
            'corrected_predictions': self.corrected_predictions,
            'time_to_first_detection': self.time_to_first_detection,
            'invocation_times_count': self.invocation_times_count
        }

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            pickle.dump(metrics_data, f)

    def load_metrics(self, filename: str):
        """Load all metrics data from a file."""
        with open(filename, 'rb') as f:
            metrics_data = pickle.load(f)
        
        self.true_positives = defaultdict(int, metrics_data['true_positives'])
        self.true_negatives = defaultdict(int, metrics_data['true_negatives'])
        self.false_positives = defaultdict(int, metrics_data['false_positives'])
        self.catch_all_entities = metrics_data['catch_all_entities']
        self.false_positives_bucket = (
            defaultdict(int, metrics_data['false_positives_bucket'][0]),
            defaultdict(int, metrics_data['false_positives_bucket'][1])
        )
        self.false_negatives = defaultdict(int, metrics_data['false_negatives'])
        self.corrected_predictions = metrics_data['corrected_predictions']
        self.time_to_first_detection = metrics_data['time_to_first_detection']
        self.invocation_times_count = metrics_data['invocation_times_count']
