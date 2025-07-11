import pickle

class FlopsCalculator:
    def __init__(self, coefficients_fp: str):
        # Load coefficients from the specified file
        with open(coefficients_fp, 'rb') as f:
            coeffs = pickle.load(f)

        self.bert_coeffs = coeffs['bert_coeffs']
        self.model_1_coeffs = coeffs['model_1_coeffs']
        self.model_2_flops_per_token = coeffs['model_2_flops_per_token']

    def calculate_flops(self, model_type, token_count):
        """ Calculate the FLOPS based on the regression model and the input text.
        Args:
            model_type (str): Type of model ('ner', 'model_1', or 'model_2')
            token_count (int): The number of tokens to calculate FLOPS for.
        Returns:
            int: The total FLOPS for the input text.
        """
        if model_type == "ner":
            if self.bert_coeffs is None:
                raise ValueError("BERT coefficients not available. Make sure the calculator was properly initialized.")
            a, b, c = self.bert_coeffs
            return int(a * (token_count ** 2) + b * token_count + c)
        elif model_type == "model_1":
            if self.model_1_coeffs is None:
                raise ValueError("Model 1 coefficients not available. Make sure the calculator was properly initialized.")
            a, b, c = self.model_1_coeffs
            return int(a * (token_count ** 2) + b * token_count + c)
        elif model_type == "model_2":
            if self.model_2_flops_per_token is None:
                raise ValueError("Model 2 FLOPS per token not available. Make sure the calculator was properly initialized.")
            return int(self.model_2_flops_per_token * token_count)
        else:
            raise ValueError("Unknown model type. Use 'ner', 'model_1', or 'model_2'.")