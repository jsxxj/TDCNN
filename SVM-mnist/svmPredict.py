import sys
import os
from svmutil import *
from PIL import Image
import datetime
# load
m = svm_load_model('./model_')
sum = 0
accu0 = 0
accu1 = 0
accu2 = 0
accu3 = 0
accu4 = 0
accu5 = 0
accu6 = 0
accu7 = 0
accu8 = 0
accu9 = 0

# predict
for item in os.listdir('testPicture'):
    path = os.path.join('testPicture', item)
    if os.path.isfile(path) and path.endswith(".bmp"):
        img_org = Image.open(path)
        answer = int(item[0])
        tmpfile = open("tmpfile", "wb")
       # print answer
    #img = img_org.resize((16, 16), Image.NEAREST)
        pixdata = img_org.load()
        line = str(answer) + ' '
        for i in range(0, 784):
            line += str(i + 1)
            if pixdata[i / 28, i % 28] == 255:
                line += ":0 "
            else:
                line += ":1 "
        tmpfile.write(line + "\n")
        tmpfile.close()
        max = 100.0
        maxidx = -1
        y, x = svm_read_problem("tmpfile")

        startTime = datetime.datetime.now();
        label, acc, val = svm_predict(y, x, m)
        endTime = datetime.datetime.now();
        print endTime - startTime
        #print "val is: ", val[0][0]
#print "probably is: ", int(abs(label[0]))
        if answer == int(abs(label[0])) and answer == 0:
            accu0 += 1
        if answer == int(abs(label[0])) and answer == 1:
            accu1 += 1
        if answer == int(abs(label[0])) and answer == 2:
            accu2 += 1
        if answer == int(abs(label[0])) and answer == 3:
            accu3 += 1
        if answer == int(abs(label[0])) and answer == 4:
            accu4 += 1
        if answer == int(abs(label[0])) and answer == 5:
            accu5 += 1
        if answer == int(abs(label[0])) and answer == 6:
            accu6 += 1
        if answer == int(abs(label[0])) and answer == 7:
            accu7 += 1
        if answer == int(abs(label[0])) and answer == 8:
            accu8 += 1
        if answer == int(abs(label[0])) and answer == 9:
            accu9 += 1
    sum += 1
print '[0    1    2   3   4   5   6   7   8   9]'
print accu0 * 1.0 / 100 ,' ',accu1 * 1.0 / 100 ,' ',accu2 * 1.0 / 100 ,' ',accu3 * 1.0 / 100 ,' ',accu4 * 1.0 / 100 ,' ',accu5 * 1.0 / 100 ,' ',accu6 * 1.0 / 100 ,' ',accu7 * 1.0 / 100 ,' ',accu8 * 1.0 / 100 ,' ',accu9 * 1.0 / 100 ,' '
print (accu0 + accu1 + accu2 + accu3 + accu4 + accu5 + accu6 + accu7 + accu8 + accu9 ) * 1.0/sum
