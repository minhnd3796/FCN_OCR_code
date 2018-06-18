import numpy as np

def valid(box):
    """
    Checks if a box is valid or not
    box: a numpy array with upper-left and lower-right point box[x1, y1, x2, y2]
    """
    return box[0] < box[2] and box[1] < box[3]


def overlap(box1, box2):
    """
    Returns the overlapped box of two boxes if the the result is valid
    """
    result = np.array([max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])])
    return result if valid(result) else None


def area(box):
    """
    Returns the area of the box
    """
    return (box[2] - box[0]) * (box[1] - box[3])

def is_close(a, b, margin=0.9):
    _overlap = overlap(a, b)
    return False if _overlap is None else area(_overlap) / max(area(a), area(b)) >= margin


# preds, truth = np.array(size=(num_boxes, 4))
def compare(preds, truth):
    hits = np.zeros(len(truth), dtype=int)
    for i, a in enumerate(truth):
        for b in preds:
            if is_close(a, b):
                hits[i] += 1
    correct = (hits == 1).all() and len(preds) == len(truth)
    recall = (hits >= 1).all()
    return correct, recall, hits

""" def pad(box, margins):
    return np.array([box[0] - margins[0], box[1] - margins[1], box[2] + margins[2], box[3] + margins[3]])
def crop(image, box):
    return image[box[1]:box[3], box[0]:box[2]] """

