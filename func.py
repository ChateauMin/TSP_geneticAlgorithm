
import random
import operator
from operator import itemgetter
import numpy as np
import csv
import decimal

# not organized with class

# New function with identical name for TSP.csv

'''
class Route:
    def __init__(self, order, length, fitness):
        self.order = order
        self.length = length
        self.fitness = fitness
    def testPrint(self):
        print("order : "+str(self.order))
        print("length : "+str(self.length))
        print("fitness : "+str(self.fitness))
'''


class geneInfo:
    def __init__(self, order, length, fitness):
        self.order = order
        self.length = length
        self.fitness = fitness

    def testPrint(self):
        print("order : "+str(self.order))
        print("\nlength : "+str(self.length))
        print("fitness : "+str(self.fitness))


def generateGene(geneElementCount):  # gene is constructed with idx
    # input idx 0~N-1 total input :N
    # range (1,N) >>1 ~ N-1 cuz TSP must starts from city '0'
    citys = [i for i in range(1, geneElementCount)]
    random.shuffle(citys)
    return citys


def calDistance(list):  # input : list containing 1~999
    # print("list[0] : "+str(list[0]))  ##### test code######
    # travel city'0'~ city'list[0]'
    distance = getDistanceList(0, list[0])
    # print("distance: "+str(distance))
    for i in range(len(list)-2):  # idx=0~998  len(list)==999
        distance = distance + getDistanceList(list[i], list[i+1])
    distance = distance + getDistanceList(list[len(list)-1], 0)
    return distance


def getDistanceList(former, latter):

    # print("former : "+str(former)) ######test code########33
    # print("latter "+str(latter))
    dist = float(totalDistance[former][latter])
    return dist


def getDistance(former, latter):
    #print("getDis activate") ######test code########
    # print("former : "+str(former))
    # print("latter : "+str(latter))
    city1 = [float(cities[former][0]), float(cities[former][1])]
    city2 = [float(cities[latter][0]), float(cities[latter][1])]
    dist = np.linalg.norm(np.array(city1)-np.array(city2))
    # print(dist)  ####### test code######3
    return dist


def sendTotalDistanceList(totalDistanceList):
    global totalDistance  # make totalDistance global
    totalDistance = totalDistanceList
    totalDistance = tuple(totalDistance)


def sendCityList(cityList):
    global cities  # make cites global in file : func
    cities = cityList
    cities = tuple(cities)


def sendNearbyCityList(nearbyCityList):
    global nearbyCities  # make nearbyCity global in file : func
    nearbyCities = nearbyCityList
    nearbyCities = tuple(nearbyCities)


def sendSearchPressure(pressure):
    global searchPressure
    searchPressure = pressure


def gendGenerationSpan(generationSpan, popSize):
    global generation
    generation = generationSpan
    global currentGenerationLevel
    currentGenerationLevel = 0
    global populationSize
    populationSize = popSize


'''##mini TSP###
def generateGene():  # TSP complete
    citys = ["B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    random.shuffle(citys)
    return citys
def calDistance(list):  # TSP complete
    distance = data.distance_A[list[0]]
    for i in range(len(list)-1):
        distance = distance + getDistance(list[i], list[i+1])
    distance = distance + getDistance(list[len(list)-1], "A")
    return distance
def getDistance(former, latter):  # TSP complete
    if former == "A":
        return data.distance_A[latter]
    elif former == "B":
        return data.distance_B[latter]
    elif former == "C":
        return data.distance_C[latter]
    elif former == "D":
        return data.distance_D[latter]
    elif former == "E":
        return data.distance_E[latter]
    elif former == "F":
        return data.distance_F[latter]
    elif former == "G":
        return data.distance_G[latter]
    elif former == "H":
        return data.distance_H[latter]
    elif former == "I":
        return data.distance_I[latter]
    elif former == "J":
        return data.distance_J[latter]
    elif former == "K":
        return data.distance_K[latter]
'''


def calFitness(geneList):
    distanceList = []
    worstDistance = 0
    bestDistance = 0
    # choose k : 3 or 4  higher the k stronger the select pressure for fitness
    k = searchPressure

    for j in range(len(geneList)):
        distanceList.append(geneList[j].length)

    worstDistance = max(distanceList)
    bestDistance = min(distanceList)

    if worstDistance-bestDistance > 100:
        for j in range(len(geneList)):
            geneList[j].fitness = float(
                (worstDistance-geneList[j].length)/10)**k  # + ((worstDistance-bestDistance)/(k-1)))
    else:
        for j in range(len(geneList)):
            geneList[j].fitness = float(
                (worstDistance-geneList[j].length))**k  # + ((worstDistance-bestDistance)/(k-1)))

    return geneList


def topFitness(genes, superiorCount):
    superiorGenes = []
    genes.sort(key=operator.attrgetter('length'))
    superiorGenes = genes[:superiorCount]  # most superior genes
    # del genes[:superiorCount] # delete genes already selected

    return superiorGenes


def wheelRoulette(genes):
    totalFitness = float(sum(gene.fitness for gene in genes))
    if totalFitness != 0:
        portions = [gene.fitness/totalFitness for gene in genes]
        selectedGene = random.choices(genes, weights=portions, k=1)
    else:
        return -1  # totalFitness==0 local optimized

    return selectedGene[0]  # selectedGene[0] cuz random.choices returns list
    ''' # unproper wheelRoulette
    pick = random.uniform(0, max)
    current = 0
    for gene in genes:
        current += gene.fitness
        if current > pick:
            return gene
    '''


def sortGene(genes, geneCount):
    # superiorCount = 10
    i = 0
    # candidates=topFitness(genes,superiorCount)
    candidates = []
    while i < geneCount:
        # if use superior genes-superiorCount
        sortedGene = wheelRoulette(genes)
        if sortedGene == -1:
            return -1  # all genes identical local optimized
        if i == 0:
            candidates.append(sortedGene)
        elif overlapKiller(candidates, sortedGene):
            candidates.append(sortedGene)
        else:
            i = i-1  # if overlap : resort gene

        i = i+1

    return candidates


def overlapKiller(candidates, newcomer):
    for gene in candidates:
        if gene == newcomer:
            # print("overlap detected")
            return False
    return True


def calDistanceCrossover(geneList):
    distance = 0.0
    for i in range(len(geneList)-2):
        distance = distance + getDistanceList(geneList[i], geneList[i+1])
    return distance


def lostGeneFinder(parent, offspring):

    totalGenes = parent.copy()
    presentGenes = set(offspring.copy())
    '''
    #####TEST CODE######
    print("totalGenes : ", end=' ')
    print(len(totalGenes))
    print("presentGenes : ", end=' ')
    print(len(presentGenes))
    '''
    for gene in parent:
        if gene in presentGenes:
            totalGenes.remove(gene)
            presentGenes.remove(gene)
    '''
    #####TEST CODE######
    print("lostGenes : ", end=' ')
    print(len(totalGenes))
    '''
    # Eventually totalGenes contains lost Genes
    # returns tuple
    return totalGenes


def lostGeneRegenerator(lostGenes, offspring):
    for gene in lostGenes:
        headDistance = getDistanceList(gene, offspring[0])
        tailDistance = getDistanceList(offspring[len(offspring)-1], gene)
        if headDistance > tailDistance:
            offspring.append(gene)
        else:
            offspring.insert(0, gene)

    return offspring


def offspringGeneCreator(parent, child):
    for i in range(len(child)):
        lostGenes = lostGeneFinder(parent[i].order, child[i])
        child[i] = lostGeneRegenerator(lostGenes, child[i])

    return child


def crossover(newGenerationGeneList, parent):

    while True:
        points = random.sample(range(0, len(cities)-3), 2)
        if abs(points[0]-points[1]) > 1:  # MUST slice each genes into 3 pieces
            points.sort()
            break
    # slice parent node
    paternalHead = (parent[0].order).copy()[:(points[0]+1)]
    maternalHead = (parent[1].order).copy()[:(points[0]+1)]

    paternalBody = (parent[0].order).copy()[(points[0]+1):(points[1]+1)]
    maternalBody = (parent[1].order).copy()[(points[0]+1):(points[1]+1)]

    paternalLeg = (parent[0].order).copy()[(points[1]+1):]
    maternalLeg = (parent[1].order).copy()[(points[1]+1):]

    paternalBodyDistnace = calDistanceCrossover(paternalBody)
    maternalBodyDistance = calDistanceCrossover(maternalBody)

    # inherit body which has least distance
    if paternalBodyDistnace >= maternalBodyDistance:
        finalBody = maternalBody
    else:
        finalBody = paternalBody
    '''
    if len(paternalHead)+len(finalBody)+len(paternalLeg) != 999:
        print("total len mismatch")
        print(len(paternalHead)+len(paternalBody)+len(paternalLeg))
        exit()
    '''
    child1 = []
    child2 = []

    child1.extend(paternalHead)
    child1.extend(finalBody)
    child1.extend(maternalLeg)

    child2.extend(maternalHead)
    child2.extend(finalBody)
    child2.extend(paternalLeg)
    '''
    if len(child1) != 999 or len(child2) != 999:
        print("child1 or 2 len mismatch")
        print(len(child1))
        print(len(child2))
        print(child1)
        exit()
    '''
    # delete duplicated elements
    child1 = list(dict.fromkeys(child1))
    child2 = list(dict.fromkeys(child2))
    '''
    if len(child1) != 999 or len(child2) != 999:
        print(len(child1))
        print(len(child2))
    '''
    offspringGene = []
    offspringGene.append(child1)
    offspringGene.append(child2)
    '''
    if len(offspringGene) != 2:
        print("childGene [][] len mismatch")
    '''
    offspringGene = offspringGeneCreator(parent, offspringGene)
    '''
    if len(offspringGene[0]) > 999 or len(offspringGene[1]) > 999:
        print("child len overflow")
        print(len(offspringGene[0]))
        print(len(offspringGene[0]))
        exit()
    '''
    # mutation Level according to DHM/ILC 100%->0%
    mutationRate = 1.0-(currentGenerationLevel/generation)
    for gene in offspringGene:

        if random.uniform(0.0, 1.0) <= mutationRate:
            while True:
                tmpGene = gene.copy()
                tmpGene = mutation(tmpGene)
                if overlapKiller(newGenerationGeneList, tmpGene):
                    gene = tmpGene
                    break

    return offspringGene


''' # this crossover is too monotonous
    turn = random.randint(0, 1)  # choose whose gene is base
    # number of genes inherit to child from base
    # select base gene where to cut and inherit by random
    geneNum = random.randint(1, len(cities)-2)
    if turn == 0:
        child = mixGene(parent[0], (parent[1].order), geneNum)  # base dad
    else:
        child = mixGene(parent[1], parent[0], geneNum)  # base mom
    return child
    '''

''' #This function is no longer used cuz of change in crossover function
def mixGene(baseGene, sourceGene, geneNum):
    # pickedGene= random.sample(baseGene.order,geneNum)
    # pickedGene.sort()  #unproper for TSP
    mutationRate = 20  # 1/mutationRate
    # newGene = data.Route([0], 0, 0)
    newGene = geneInfo([0], 0, 0)
    base = baseGene.order[:geneNum]
    source = sourceGene.order[:]
    source = sourceGeneInspector(source, base)
    newGene.order = base+source
    if 1 == random.randint(1, mutationRate):  # mutation
        newGene.order = mutation(newGene.order)
    newGene.length = calDistance(newGene.order)
    # newGene.fitness = calFitness(newGene.length) inproper fitness control
    return newGene
'''


def sourceGeneInspector(sourceGene, pickedGene):
    # print("source: "+str(sourceGene)) # test code error fix
    # print("picked: "+str(pickedGene))
    for i in range(len(pickedGene)):
        sourceGene.remove(pickedGene[i])
    return sourceGene


def mutation(gene):

    mutationResult = []
    mutationGene = []
    mutationDistance = []

    slideGene = gene.copy()
    slideGene = slideMutation(slideGene)
    mutationGene.append(slideGene)
    slideGeneDistance = calDistance(slideGene)
    mutationDistance.append(slideGeneDistance)

    inversionGene = gene.copy()
    inversionGene = inversionMutation(inversionGene)
    mutationGene.append(inversionGene)
    inversionGeneDistance = calDistance(inversionGene)
    mutationDistance.append(inversionGeneDistance)

    irgibnnmGene = gene.copy()
    irgibnnmGene = irgibnnmMutation(irgibnnmGene)
    mutationGene.append(irgibnnmGene)
    irgibnnmGeneDistance = calDistance(irgibnnmGene)
    mutationDistance.append(irgibnnmGeneDistance)

    mutationResult.append(
        sorted(zip(mutationGene, mutationDistance), key=itemgetter(1)))
    ''' ##TEST CODE###
    print("total mutatuion")
    print(mutationResult)
    print("mutation Result: ")
    '''
    gene = mutationResult[0][0][0]  # return best mutation gene
    # print(mutationResult[0][0][0])
    # exit()

    '''#unproper mutation
    mutationLevel = random.randint(len(gene)//2, len(gene))
    for i in range(mutationLevel):  # pick two idx and swap
        exchange = random.sample(range(0, len(gene)), 2)
        gene[exchange[0]], gene[exchange[1]
                                ] = gene[exchange[1]], gene[exchange[0]]
                                '''
    return gene


def chooseTwoGeneElement(gene, geneIndex):
    interval = 0
    while True:
        geneIndex = random.sample(range(0, len(gene)), 2)
        geneIndex.sort()
        # do not allow index right next to each other
        interval = abs(geneIndex[0]-geneIndex[1])-1
        if interval > 0:
            break
    return geneIndex


def slideMutation(gene):
    geneIndex = []
    geneIndex = chooseTwoGeneElement(gene, geneIndex)
    interval = abs(geneIndex[0]-geneIndex[1])-1
    tmp = gene[geneIndex[0]+1]  # idx 2nd will slide
    for i in range(interval):
        # print(str(gene[(geneIndex[0]+1)+(i+1)]) +
        #      " to position : "+str(gene[(geneIndex[0]+1)+i])) #####TEST CODE#####
        gene[(geneIndex[0]+1)+i] = gene[(geneIndex[0]+1)+(i+1)]

    gene[geneIndex[1]] = tmp
    return gene


def inversionMutation(gene):
    geneIndex = []
    geneIndex = chooseTwoGeneElement(gene, geneIndex)
    interval = abs(geneIndex[0]-geneIndex[1])-1
    swapCount = int(interval//2)+1
    for i in range(swapCount):  # mirror genes
        gene[(geneIndex[0]+1)+i], gene[geneIndex[1]-i
                                       ] = gene[geneIndex[1]-i], gene[(geneIndex[0]+1)+i]
       # print("swap : "+str(gene[(geneIndex[0]+1)+i]) +
       #       " and "+str(gene[geneIndex[1]-i]))#####TEST CODE#######3
    return gene


def irgibnnmMutation(gene):
    gene = inversionMutation(gene)
    # choose single gene excpt city 0
    randomGene = random.randint(1, len(gene))  # city1~[len(gene)]
    # nearbyCities is [idx list] of nearest cities
    # nearest city from randomGene
    randomGeneClosest = int(nearbyCities[randomGene][0])
    # choose city which is around randomGeneClosest ()
    while True:
        nearbycity = int(random.choice(nearbyCities[randomGeneClosest][:5]))
        if nearbycity != randomGene and nearbycity != randomGeneClosest:
            break
    gene[randomGene-1], gene[nearbycity -
                             1] = gene[nearbycity-1], gene[randomGene-1]
    # idx randomGene-1,nearbycity-1  cuz randomGene,nearbycity are citynumber not idx number
    return gene


def newGeneration(geneList, geneCount):
    # DHM/ILC crossover does not consider survivor gene
    #geneCount = geneCount - len(geneList)
    newGenerationGeneList = []
    newGenerationList = []

    for i in range(geneCount//2):
        parent = sortGene(geneList, 2)
        if parent == -1:
            return -1  # all genes identical local optimized
        child = crossover(newGenerationGeneList, parent)
        for gene in child:
            newGenerationGeneList.append(gene)

    # calculate fitness
    for gene in newGenerationGeneList:
        distance = calDistance(gene)
        newGenerationList.append(geneInfo(gene, distance, 0.0))

    newGenerationList = calFitness(newGenerationList)
    # newGenerationList = newGenerationList+geneList ## survivor is no more
    return newGenerationList


def getBestGene(gene):
    bestGene = gene[0]
    for i in range(len(gene)-2):
        if gene[i].fitness < gene[i+1].fitness:
            bestGene = gene[i+1]
    return bestGene


def getWorstGene(gene):
    worstGene = gene[0]
    for i in range(len(gene)-2):
        if gene[i].fitness > gene[i+1].fitness:
            worstGene = gene[i+1]
    return worstGene


def writeCSV(writer, bestGene):

    length = []
    length.append(bestGene.length)
    city = []

    gen = []
    gen.append(generation)
    popSize = []
    popSize.append(populationSize)
    pressure = []
    pressure.append(searchPressure)

    writer.writerow(length)
    writer.writerow(';')

    writer.writerow(bestGene.order)
    writer.writerow(';')
    writer.writerow(gen)
    writer.writerow(';')
    writer.writerow(popSize)
    writer.writerow(';')
    writer.writerow(pressure)
    writer.writerow(';;;')