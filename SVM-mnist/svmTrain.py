import sys
from svmutil import *
from PIL import Image
import random
import datetime


y, x = svm_read_problem('trainData')
#  if i == 4 or i == 3:
#    m = svm_train(y, x, '-c 10000')
#  else:
startTime = datetime.datetime.now();
m = svm_train(y, x, '-c 3 -g 0.015625')
endTime = datetime.datetime.now();
print endTime - startTime
svm_save_model('./model_', m)
