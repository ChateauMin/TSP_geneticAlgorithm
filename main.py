#!C:\Program Files\Python39\python.exe
# print("content-type: text/html; charset=utf-8\n")
# print()
import cgi
import os
import os.path
import func
import csv
import sys
import random

def recursion():
    # read TSP.csv and store it to list
    cities = []
    distancePerCity = []
    nearbyCity = []
    generation = 1600  # generation span
    generationCount = [400]
    genCount = random.choice(generationCount)  # number of population
    pressure = [2.9, 3.0, 3.1]
    searchPressure = random.choice(pressure) # pressure for fitness
    print(genCount, searchPressure)

    print("Reading TSP.csv ...")
    with open('TSP.csv', mode='r', newline='') as tsp:
        reader = csv.reader(tsp)
        # i = 0
        for row in reader:
            cities.append(row)
        '''  if i > 50:
                break
            i = i+1'''
    print("TSP.csv Read complete")

    print("Reading totalDistance.csv ...")
    with open('totalDistance.csv', mode='r', newline='') as allDistance:
        reader = csv.reader(allDistance)
        for row in reader:
            distancePerCity.append(row)
    print("totalDistance.csv Read complete")

    print("Reading cityDistance.csv ...")
    with open('cityDistance.csv', mode='r', newline='') as cityDistance:
        reader = csv.reader(cityDistance)
        for row in reader:
            nearbyCity.append(row)
    print("cityDistance.csv Read complete")

    func.sendCityList(cities)
    func.sendTotalDistanceList(distancePerCity)
    func.sendNearbyCityList(nearbyCity)
    func.sendSearchPressure(searchPressure)
    func.gendGenerationSpan(generation, genCount)

    genes = []  # store genelist
    gen = []  # city travel order for single gen
    distance = 0.0  # total distance of single gene
    fitness = 0.0  # fitness of single gene
    # survivors = 200  # number of genes survived in single generation
    # DHM/ILC crossover does not need survivor gene
    survivorGenes = []  # array of genes survived in one generation


    '''
    # generate gene and calculate fitness, distance
    for j in range(genCount):
        gen = func.generateGene()
        distance = func.calDistance(gen)
        fitness = func.calFitness(distance)
        genes.append(func.TSP(gen, distance, fitness))
    '''
    # TSP.csv type version
    print("creating 0st Gen ...")
    for j in range(genCount):
        gen = func.generateGene(len(cities))
        distance = func.calDistance(gen)
        # fitness = func.calFitness(distance) # inproper fitness calculation
        genes.append(func.geneInfo(gen, distance, fitness))
        # fitness: yet calculated

    # calculate fitness
    func.calFitness(genes)

    bestGene = func.geneInfo([0], 999999, 0)
    identicalCount = 0  # same result count
    recordGuardCount = 0  # best record guard count
    breakCount = 9
    newRecord = 0
    bestRecord = 0
    localOptimized = 0  # check whether localOptimized

    for j in range(generation):
        func.currentGenerationLevel = func.currentGenerationLevel+1
        # sort genes with wheelRoulette : count survivors
        # !!## survivorGenes = func.sortGene(genes, survivors)
        # DHM/ILC crossover does not need survivor gene
        # create new generation
        genes = func.newGeneration(genes, genCount)
        if genes == -1:  # all genes identical local optimaized
            localOptimized = 1
            break

        newGene = func.getBestGene(genes)
        newWorstGene = func.getWorstGene(genes)
        newRecord = newGene.fitness

        # Test print code
        if newGene.length != bestGene.length:
            print("================ "+str(j+1) +
                "th Gen ===========================")
            newGene.testPrint()
            if j > 0:  # do not print at 1st gen
                print("\nWORST gene length : "+str(newWorstGene.length)+"\n")
                print("PREVIOUS BEST RECORD : "+str(bestGene.length))
            identicalCount = 0
        else:
            identicalCount = identicalCount+1
            print("identical count : "+str(identicalCount))

        if newGene.length < bestGene.length:
            bestGene = newGene
            recordGuardCount = 0
            i = j+1
        else:
            recordGuardCount = recordGuardCount+1
            print("Record Guard Count : "+str(recordGuardCount))
        print("mutaionRate:", end=' ')
        print(1.0-(func.currentGenerationLevel/generation))
        print('previous best generation : '+str(i))
        print('present generation : '+str(j+1))
        # loop until gen span ends
        #
        if identicalCount > breakCount:
            break

        print("\ncreating next gen...\n")


    print("===================Final Gen=====================")
    bestGene.testPrint()

    if localOptimized == 1:
        print("local optimized >> BREAK\n")

    if identicalCount > breakCount:
        print("BREAK : best result identical over " + str(identicalCount)+" times")


    print("\n saving data in previousReult.csv ...")
    with open('previousResults.csv', 'a', newline='') as storeResult:
        writer = csv.writer(storeResult)

        # wrtie previous records
        # order : length-> gene order-> fitness-> generation -> gencount->searchPressure
        func.writeCSV(writer, bestGene)
    print("complete saving data in previousReult.csv \n")

    print("checking previous best data...\n")


    if os.path.isfile('bestResults.csv'):
        with open('bestResults.csv', 'r', newline='') as readBestRecord:
            reader = csv.reader(readBestRecord)
            for row in reader:
                if row[0] != ';':
                    previousFileRecord = float(row[0])
                    print("previous record : "+str(previousFileRecord))
                    break
    else:
        previousFileRecord = 99999.0


    if previousFileRecord > bestGene.length:
        print("best record renewed..")
        with open('bestResults.csv', 'w+', newline='') as overWriteResult:
            writer = csv.writer(overWriteResult)
            func.writeCSV(writer, bestGene)
            print("new record saved in bestResults.csv\n")
    else:
        print("Record guarded changes : NONE\n")

    '''
        bestGene.append(CandidateGene)
        result=func.getBestGene(bestGene)
        result.testPrint()
    '''


    ''' #survivor test code
    print("SURVIVORS")
    for gene in survivorGenes:
        gene.testPrint()
    '''
    recursion()

recursion()
