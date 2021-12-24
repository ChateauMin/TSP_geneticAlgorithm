import os
import os.path
import func
import csv

information = []
breakCount = 0
distance = 0

if os.path.isfile('bestResults.csv'):
    print("Reading bestResult.csv...\n")
    with open('bestResults.csv', mode='r', newline='') as result:
        reader = csv.reader(result)
        # i = 0
        for info in reader:
            if info[0] != ';':
                information.append(info)
            else:
                continue
else:
    print("ERROR : bestResult.csv not Found")

print("order : ", end='')
print(information[1])
print("\n length : "+str(information[0][0]))
print("generation span : "+str(information[2][0]))
print("population size per generation : "+str(information[3][0]))
print("search pressure : "+str(information[4][0]))

print("\n\n generating solution.csv... \n ")


with open('solution.csv', 'w+', newline='') as solution:
    writer = csv.writer(solution)
    cityNumber = []
    writer.writerow('0')  # add city:0 first

    for city in information[1]:
        cityNumber.append(int(city))
        writer.writerow(cityNumber)
        cityNumber.clear()
    print("solution.csv Generate completed\n")
