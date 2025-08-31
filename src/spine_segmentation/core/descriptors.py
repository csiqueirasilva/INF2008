import numpy as np
import cv2

def hog_desc(img):
    """
    Return a HOG descriptor for a 384x384 grayscale uint8 image.
    We ensure dtype, shape, and contiguity before calling OpenCV.
    """
    # to gray
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]
    # resize to the window size we defined
    if img.shape[:2] != (384, 384):
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)
    # enforce dtype + contiguity
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    hog = cv2.HOGDescriptor(_winSize=(384,384),
                            _blockSize=(32,32),
                            _blockStride=(16,16),
                            _cellSize=(16,16),
                            _nbins=9)
    v = hog.compute(img).reshape(-1).astype(np.float32)
    return v

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    den = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / den)
