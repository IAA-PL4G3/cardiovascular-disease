import numpy as np
from scipy.special import expit

def analyze_confidence(predictions, probabilities, y_test, model_name="Model"):
    # takes predictions, probabilities, and true labels to analyze confidence on hits and misses
    hits = []
    misses = []
    
    for i, pred in enumerate(predictions):
        true_label = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]

        # extract confidence score
        if isinstance(probabilities[0], np.ndarray) or len(probabilities.shape) > 1:
            confidence = probabilities[i][pred]
        else:
            confidence = probabilities[i] if pred == 1 else 1 - probabilities[i]
        
        if pred == true_label:
            hits.append(confidence)
        else:
            misses.append(confidence)
    
    # calculate statistics
    stats = {
        "model_name": model_name,
        "hits": {
            "values": hits,
            "avg": np.mean(hits) if hits else 0,
            "max": np.max(hits) if hits else 0,
            "min": np.min(hits) if hits else 0,
            "count": len(hits)
        },
        "misses": {
            "values": misses,
            "avg": np.mean(misses) if misses else 0,
            "max": np.max(misses) if misses else 0,
            "min": np.min(misses) if misses else 0,
            "count": len(misses)
        }
    }
    
    return stats

def print_confidence_report(stats):
    # print a report of confidence statistics for hits and misses
    model_name = stats["model_name"]
    hits = stats["hits"]
    misses = stats["misses"]
    
    print(f"{model_name} Confidence on Hits:")
    print(f"  Average: {hits['avg']:.4f}, Max: {hits['max']:.4f}, Min: {hits['min']:.4f}")
    print(f"  Count: {hits['count']}")
    print(f"{model_name} Confidence on Misses:")
    print(f"  Average: {misses['avg']:.4f}, Max: {misses['max']:.4f}, Min: {misses['min']:.4f}")
    print(f"  Count: {misses['count']}\n")

def get_index_from_stats(stats, value_type, score_type):
    # helper function to get index of a specific score type (avg, max, min) from hits or misses
    values = stats[value_type]["values"]
    target_value = stats[value_type][score_type]
    return values.index(target_value) if isinstance(values, list) else next(
        i for i, v in enumerate(values) if v == target_value
    )