import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from postprocess.registration import RegistrationOcr, parse_registration_address, extract_registration_address

image = cv2.imread(sys.argv[1])
ocr = RegistrationOcr(gpu=True)
idx, text = ocr.read(image)
print(f"Лучший поворот (idx 0..3): {idx}")
print(f"Сырой OCR-текст:\n  {text}\n")
addr = parse_registration_address(text)
print(f"Разобранный адрес: {addr}")
field = extract_registration_address(image, ocr)
print(f"RefinedField: {field}")
