"""
Utility functions for video summarization evaluation
"""

def get_segments_from_indices(indices):
    if not indices:
        return []

    indices = sorted(set(indices))
    segments, start = [], indices[0]

    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            segments.append((start, indices[i - 1]))
            start = indices[i]

    segments.append((start, indices[-1]))
    return segments


def expand_segment(seg):
    return list(range(seg[0], seg[1] + 1))


def calculate_segment_f1(user_segments, model_segments):
    tp = 0

    for m_seg in model_segments:
        m_range = set(expand_segment(m_seg))
        for u_seg in user_segments:
            if m_range & set(expand_segment(u_seg)):
                tp += 1
                break

    precision = tp / len(model_segments) if model_segments else 0
    recall = tp / len(user_segments) if user_segments else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return f1, precision, recall
