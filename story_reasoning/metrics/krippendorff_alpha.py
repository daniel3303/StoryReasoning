import numpy as np


def krippendorff_alpha(data, level_of_measurement='nominal'):
    """
    Calculate Krippendorff's alpha for reliability of coding.
    
    Args:
        data: Data matrix with each row representing a coder and each column an item.
             Missing values should be represented as np.nan.
        level_of_measurement: The level of measurement:
             'nominal' - data are unordered categories
             'ordinal' - data are ordered ranks
             'interval' - data are measured on an interval scale
             'ratio' - data are measured on a ratio scale
        
    Returns:
        Krippendorff's alpha coefficient
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Data must be a 2D array/matrix")

    if len(data) < 2:  # Need at least 2 coders
        return 0

    # Create coincidence matrix
    coincidence_matrix, value_domain, value_counts = _create_coincidence_matrix(data)

    if coincidence_matrix is None:  # No data to compare
        return 0

    # Calculate the distance metric based on level of measurement
    delta_metric = _calculate_distance_metric(value_domain, value_counts, level_of_measurement)

    # Calculate observed disagreement
    do = _calculate_observed_disagreement(coincidence_matrix, delta_metric)

    # Calculate expected disagreement
    de = _calculate_expected_disagreement(value_counts, delta_metric)

    # Calculate alpha
    epsilon = 1e-10  # Small value to prevent division by zero
    if abs(de) < epsilon:
        return 1.0 if abs(do) < epsilon else 0.0

    alpha = 1 - (do / de)
    return alpha


def _create_coincidence_matrix(data):
    """
    Create the coincidence matrix from the data.
    
    Returns:
        coincidence_matrix: Coincidence matrix
        value_domain: Unique values in the data (sorted for ordinal, interval, ratio)
        value_counts: Counts of each value in the value domain
    """
    n_coders, n_items = data.shape

    # Get unique values (excluding NaN)
    value_domain = np.unique(data[~np.isnan(data)])
    if len(value_domain) < 2:
        return None, None, None  # Not enough unique values

    n_values = len(value_domain)

    # Create a mapping from values to indices
    value_map = {val: i for i, val in enumerate(value_domain)}

    # Initialize coincidence matrix and value counts
    coincidence_matrix = np.zeros((n_values, n_values))
    value_counts = np.zeros(n_values)

    # Fill coincidence matrix
    for item in range(n_items):
        # Get values for this item (excluding NaN)
        item_values = data[:, item]
        valid_mask = ~np.isnan(item_values)
        valid_values = item_values[valid_mask]
        n_valid = len(valid_values)

        if n_valid < 2:  # Need at least 2 values for a comparison
            continue

        # Add to value counts
        for val in valid_values:
            idx = value_map[val]
            value_counts[idx] += 1

        # Add to coincidence matrix
        for i, val_i in enumerate(valid_values):
            idx_i = value_map[val_i]
            for j, val_j in enumerate(valid_values):
                if i != j:  # Don't compare value with itself
                    idx_j = value_map[val_j]
                    # Each pairing contributes 1/(n-1) to the coincidence matrix
                    coincidence_matrix[idx_i, idx_j] += 1.0 / (n_valid - 1)

    return coincidence_matrix, value_domain, value_counts


def _calculate_distance_metric(value_domain, value_counts, level_of_measurement):
    """
    Calculate the distance metric based on the level of measurement.
    """
    n_values = len(value_domain)
    metric = np.zeros((n_values, n_values))

    if level_of_measurement == 'nominal':
        # For nominal data, distance is binary (0 if same, 1 if different)
        for i in range(n_values):
            for j in range(n_values):
                metric[i, j] = 0.0 if i == j else 1.0

    elif level_of_measurement == 'ordinal':
        # For ordinal data, we need cumulative frequencies
        cumulative_counts = np.zeros(n_values)
        total_count = np.sum(value_counts)

        # Calculate cumulative counts
        cumsum = 0
        for i in range(n_values):
            cumsum += value_counts[i]
            cumulative_counts[i] = cumsum - value_counts[i]/2

        # Normalize
        cumulative_counts /= total_count

        # Calculate distances
        for i in range(n_values):
            for j in range(n_values):
                metric[i, j] = (cumulative_counts[i] - cumulative_counts[j])**2

    elif level_of_measurement in ['interval', 'ratio']:
        # For interval and ratio data, the distance is the squared difference
        for i in range(n_values):
            for j in range(n_values):
                metric[i, j] = (value_domain[i] - value_domain[j])**2

    else:
        raise ValueError(f"Unknown level of measurement: {level_of_measurement}")

    return metric


def _calculate_observed_disagreement(coincidence_matrix, delta_metric):
    """
    Calculate the observed disagreement using the coincidence matrix and distance metric.
    """
    n = coincidence_matrix.shape[0]
    total_coincidences = np.sum(coincidence_matrix)

    if total_coincidences == 0:
        return 0

    observed_disagreement = 0
    for i in range(n):
        for j in range(n):
            observed_disagreement += coincidence_matrix[i, j] * delta_metric[i, j]

    return observed_disagreement / total_coincidences


def _calculate_expected_disagreement(value_counts, delta_metric):
    """
    Calculate the expected disagreement using value counts and distance metric.
    """
    n = len(value_counts)
    total_count = np.sum(value_counts)

    if total_count <= 1:
        return 0

    expected_disagreement = 0
    for i in range(n):
        for j in range(n):
            expected_disagreement += value_counts[i] * value_counts[j] * delta_metric[i, j]

    return expected_disagreement / (total_count * (total_count - 1))