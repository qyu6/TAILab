'''
@date:2021.10.16
@language: eng: english; chi_sim: Chinese Simple
'''

def ocr(img_path,language):
    import cv2
    import numpy as np
    import pytesseract
    from PIL import Image

    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.png", img)

    #  Apply threshold to get image with only black and white
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Write the image after apply opencv to do some ...
    cv2.imwrite(img_path, img)

    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(img_path),lang=language)
    print(result)
    # Remove template file

    return result


# print('--- Start recognize text from image ---')
# print(ocr('test1.png'))
# print("------ Done -------")
# ocr('pics/ocr_test.png','chi_sim')