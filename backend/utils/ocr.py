import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import cv2


def preprocess_image(pil_img):
    img = np.array(pil_img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise
    gray = cv2.medianBlur(gray, 3)

    # Threshold (important for text clarity)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def extract_text(file_bytes):
    images = convert_from_bytes(file_bytes, dpi=300)

    full_text = ""

    for i, img in enumerate(images):
        processed = preprocess_image(img)

        # OCR config tuned for documents
        text = pytesseract.image_to_string(
            processed,
            config="--oem 3 --psm 6"
        )

        full_text += f"\n\n===== PAGE {i+1} =====\n\n"
        full_text += text

    return full_text