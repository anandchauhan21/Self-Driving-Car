import utlis
from utlis import *
from sklearn.model_selection import train_test_split

### STEP 1
path = 'myData'
data = importDataInfo(path)

### STEP 2

data = balanceData(data, display=False)

### STEP 3
imagesPath, steering = utlis.loadDate(path, data)
print(imagesPath[0], steering[0])

### STEP 4
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)
print('Total Training Image: ', len(xTrain))
print('Total Validation Image: ', len(xVal))

### STEP 5

### STEP 6

### STEP 7

### STEP 8
