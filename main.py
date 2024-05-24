import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

dosya_yolu = "/Users/nezirer/PycharmProjects/aracTakip/arabalar1.jpg"
image1 = cv2.imread(dosya_yolu)

image_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


bbox, label, conf = cv.detect_common_objects(image1)
output_image = draw_bbox(image_rgb, bbox, label, conf)


araba_say = label.count('car')
plt.imshow(output_image)
plt.title(f'Tespit edilen araba sayisi: {araba_say}')
plt.show()


print("Resimdeki araba sayisi :" + str(araba_say))
