import numpy as np
import cv2

def overlay_mask_on_gray(gray_u8: np.ndarray, mask_u8: np.ndarray, alpha=0.35, color=(0,255,0)):
    # squeeze channel dims, enforce 2D
    if mask_u8.ndim > 2:
        mask_u8 = np.squeeze(mask_u8)
    if mask_u8.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask_u8.shape}")
    # match size
    if mask_u8.shape != gray_u8.shape:
        mask_u8 = cv2.resize(mask_u8, (gray_u8.shape[1], gray_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    # ensure uint8 0/255
    mask_u8 = ((mask_u8 > 0).astype(np.uint8)) * 255

    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    overlay = np.zeros_like(base)
    overlay[mask_u8.astype(bool)] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0.0)

def save_u8_gray(path, img):
    import numpy as np, cv2
    img = np.squeeze(img)
    if img.ndim == 3 and img.shape[0] in (1,3,4) and img.shape[-1] not in (1,3,4):
        # channels-first -> move to channels-last
        img = np.transpose(img, (1,2,0))
    if img.ndim == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.shape[2] in (3,4):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2]==3 else cv2.COLOR_BGRA2GRAY)
        else:
            img = img[:, :, 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), np.ascontiguousarray(img))