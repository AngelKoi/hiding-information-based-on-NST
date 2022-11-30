import cv2

content_list_png = []

for i in range(100):
    imagename = "single_content/"+"images" + "%d" % i + ".jpg"
    content_list_png.append(imagename)

for image in content_list_png:
    image_1 = cv2.imread(image)
    print(image_1)
    image_1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2RGB)
    cv2.imwrite(image,image_1)