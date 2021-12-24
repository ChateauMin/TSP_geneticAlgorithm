import csv
import func
from operator import itemgetter
import os
import os.path

totalDistanceList = []  # 1000x1000 list of distance
distanceList = []  # one city to other 999 cities distance list
idxList = []  # index list for closest cities
nearbyCityList = []  # list containing city idx and distance
nearbyCityIndexList = []  # list conataining only idx
cities = []  # list for reading TSP.csv
idx = list(range(1000))

with open('TSP.csv', mode='r', newline='') as tsp:
    reader = csv.reader(tsp)
    for row in reader:
        cities.append(row)

func.sendCityList(cities)


for city1 in range(len(cities)):
    for city2 in range(len(cities)):
        if city1 == city2:
            distanceList.append(-1)  # self is not considered as nearby city
        else:
            distance = func.getDistance(city1, city2)
            distanceList.append(distance)

    print("city: "+str(city1) + ">> distance calculation complete ")

    tmpCity = distanceList.copy()
    totalDistanceList.append(tmpCity)

    nearbyCityList.append(
        sorted(zip(idx, distanceList), key=itemgetter(1))[1:11])  # top 10 nearest
    distanceList.clear()

if os.path.isfile('totalDistance.csv'):
    print("totalDistance.csv exists..")
else:
    with open('totalDistance.csv', 'w', newline='') as distanceList:
        writer = csv.writer(distanceList)

        for cityDistanceList in totalDistanceList:
            writer.writerow(cityDistanceList)

i = 0
for nearList in nearbyCityList:
    for city in nearList:
        idxList.append(city[0])
    if 0 in idxList:  # remove city 0 in list cuz city 0 is always start
        idxList.remove(0)

    print("city "+str(i)+" : closest cities idx >>", end=' ')
    i = i+1
    print(idxList)
    tmpIdx = idxList.copy()
    nearbyCityIndexList.append(tmpIdx)
    idxList.clear()

# city 0 nearest city is not needed cuz city0 is always the start
nearbyCityIndexList = nearbyCityIndexList

with open('cityDistance.csv', 'w+', newline='') as cityList:
    writer = csv.writer(cityList)

    for cityIdxList in nearbyCityIndexList:
        writer.writerow(cityIdxList)
