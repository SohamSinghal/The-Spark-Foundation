import cv2
import pytesseract #Google's open source character recogniser 
#Change the path to the location where you've installed tesseract
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
#Read image
def read_image(path):
    img = cv2.imread(path)
    return img
#Convert image to grayscale
def grayscale(img):
    gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray_image
#Thresholding of image
def thresholding(img):
    thresh,t_image = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    return t_image
#Apply Median filter for denoising image
def denoise(img):
    denoise = cv2.medianBlur(img,3)
    return denoise
#Use pytesseract library
def text_extractor(img,og_image):
    text = pytesseract.image_to_string(img)
    return text
#main
def main(path):
    image = read_image(path)
    cv2.imshow("Image",image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    gray_image = grayscale(image)
    cv2.imshow("Grayscale Image",gray_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    threshold_image = thresholding(gray_image)
    cv2.imshow("Thresholding Result",threshold_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    fin_image = denoise(threshold_image)
    cv2.imshow("Final Image",fin_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    text_in_image = text_extractor(fin_image,image)
    print(text_in_image)

path = "Images\Screenshot 01.png"
main(path)

    

