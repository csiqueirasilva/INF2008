import numpy as np
import cv2

def preprocess_ct_hu(ct: np.ndarray):
    # bone window
    ct = np.clip(ct, 200, 2000)
    ct = (ct - 200) / (2000 - 200 + 1e-6)
    return ct.astype(np.float32)

def preprocess_frame(path, size=384):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.equalizeHist(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img
