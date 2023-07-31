import cv2
import matplotlib.pyplot as plt
import easyocr

# Read the images
image_path = r'C:\Users\Maryam Badarni\text_detection\data\words.jpg'
img = cv2.imread(image_path)

#Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

#Detect text on the image
text_ = reader.readtext(img)
threshold = 0.7
# Draw bbox and text

for t in text_:
    print(t)
    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0,255,0), 3)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

