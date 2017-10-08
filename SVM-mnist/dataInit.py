from PIL import Image
import os

f = open('trainData', 'wb')
#line = '-1'
for i in range(0, 10):
    for item in os.listdir(str(i)):
        path = os.path.join(str(i), item)
        if os.path.isfile(path) and path.endswith(".bmp"):
            img_org = Image.open(path)
            #img = img_org.resize((28, 28), Image.NEAREST)
            pixdata = img_org.load()
            # -1
            line = str(i)+ ' '
            for k in range(0, 784):
                line += str(k + 1)
                if pixdata[k / 28, k % 28] == 255:
                    line += ":0 "
                else:
                    line += ":1 "
            f.write(line + "\n")
            # -1
    f.close
