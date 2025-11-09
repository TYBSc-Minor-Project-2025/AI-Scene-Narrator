import pytesseract
import cv2

# Optional: manually set path if needed
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create a simple test image
import numpy as np
img = np.ones((100, 300), dtype=np.uint8) * 255
cv2.putText(img, "Hello Priii!", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0), 3)
cv2.imwrite("test_text.png", img)

# Read the text using Tesseract
text = pytesseract.image_to_string(img)
print("Detected text:", text)
