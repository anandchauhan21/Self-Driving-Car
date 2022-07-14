import utlis
from utlis import *

### STEP 1
path = 'myData'
data = importDataInfo(path)

### STEP 2

data = balanceData(data, display=False)

### STEP 3
imagesPath, steering = utlis.loadDate(path, data)
print(imagesPath[0], steering[0])
