import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
import string
from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance
from gensim.models import Word2Vec

#Data, corpora
stop_words = text.ENGLISH_STOP_WORDS

sentimentTags = pd.read_csv('Sentiment_Tags.csv')
solomonTexts = pd.read_excel('SolomonTexts.xls')
solomonSentTexts = solomonTexts[solomonTexts['Type'] == 'Sent']['Body'].tolist()

sentimentWord = sentimentTags['Word'].tolist()
sentimentScore = sentimentTags['Score'].tolist()
sentimentDict = {}
for i in range(len(sentimentWord)):
    sentimentDict[sentimentWord[i]] = sentimentScore[i]

#Helper function for computeSentenceAverages
def cleanInputData(inputText):
    sentenceList = []
    for i in range(len(inputText)):
        wordlist = inputText[i].lower().split(" ")
        cleanWordList = []
        for word in wordlist:
            #Remove any instance of punctuation in/around a word
            word =  "".join(l for l in word if l not in string.punctuation)
            cleanWordList.append(word)
        sentenceList.append(cleanWordList)

    return sentenceList

#Helper function for computeSentenceAverages: tries to correct misspellings
#Very inefficient; computes Levenshtein edit distance between candidate word and each word in our tagset
def handleMisspellings(word):
    distance = 10
    for i, referenceWord in enumerate(sentimentWord):
        editDistance = edit_distance(word, referenceWord)
        if editDistance < distance:
            closestWord, closestWordIndex = referenceWord, i
            distance = editDistance

    if distance < 4:
        score = sentimentScore[closestWordIndex]
    else:
        score  = '<unk>'
    return score

#Helper function for computeSentenceAverages: tries to find synonyms for words not in tagset
#If misspellings is True, additionally checks to see if there is a compelling misspelling candidate
#among the tagset.
def findSynonymsAndUnknowns(word, misspellings=False):
    synList = []
    #If not in our corpus make WordNet Synonyms list
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synList.append(l.name())
    #set score to 0
    score = 0
    #Test to see if any synonym words exist in our corpus
    for i in range(len(synList)):
        try:
            score = sentimentDict[synList[i]]
            break
        except KeyError:
            continue
    #Otherwise give word an unknown tag
    if score == 0:
        if misspellings:
            score = handleMisspellings(word)
        else:
            score = "<unk>"
    return score

#Main function
#Takes a list of strings as input
def computeSentenceAverages(inputText):
    cumulativeAverageScore, sentenceAverages = 0, []
    sentenceList = cleanInputData(inputText)
    individualWordScores = []

    for i in range(len(sentenceList)):
        cumulativeScore, stopWordPunctScore, unknownCount = 0, 0, 0
        wordlist = sentenceList[i]
        individualWordScores.append([])
        for word in wordlist:
            #If stop word or punctuation, skip
            if word in stop_words or word in string.punctuation:
                stopWordPunctScore += 1
                continue
            else:
                #Test to see if word is in our corpus
                try:
                    score = sentimentDict[word]
                except KeyError:
                    score = findSynonymsAndUnknowns(word)

            if score == "<unk>":
                unknownCount += 1
            else:
                cumulativeScore += score

            individualWordScores[i].append([score, word])

        #Do not count stop words or punctuation toward average
        sentLenStopWord = len(wordlist) - stopWordPunctScore - unknownCount
        if sentLenStopWord == 0:
            sentenceAverageScore = None
        else:
            sentenceAverageScore = cumulativeScore / sentLenStopWord

        if sentenceAverageScore != None:
            cumulativeAverageScore += sentenceAverageScore
            sentenceAverages.append(sentenceAverageScore)

    return [sentenceAverages, inputText, individualWordScores]

#Runs main function, compiles scores and text into desired format.
#Outputs list of Score-Sentence pairs ordered from low-scoring to high-scoring (negative to positive)
#and a list of lists of pairs with scores for individual words.
def runModel(text):
    textAvgs, textText, wordInSentenceScores = computeSentenceAverages(text)
    combinedList = []
    percentSum = 0
    for i in range(len(textAvgs)):
        if textAvgs[i] != None:
            combinedList.append([textAvgs[i], textText[i]])

    combinedList.sort()
    return [combinedList, wordInSentenceScores]

#Function for visualization
def plotScores(scores):
    ax = plt.axes()
    ax.set_ylabel("Score")
    ax.set_xlabel("Texts")
    ax.set_title("Score Distribution")
    plt.scatter(list(range(len(scores))), scores, color = 'blue', marker = 'x', label = 'score')
    plt.show()

#Takes in positivity score and returns either a binary positive/negative classification
#Or a leveled positivity rating based on the positivity dist. of Solomon's texts.
def qualitativeOutput(score, binary_leveled='leveled'):
    #Using corpus of Solomon's texts as training data to create a positivity distribution.
    #Then, compare inputted text against the training distribution
    sentimentDistribution = runModel(solomonSentTexts)
    scores = list(map(lambda x: x[0], sentimentDistribution[0]))
    trainingMean = sum(scores) / len(scores)
    trainingPlus1SD = scores[int(84*(len(scores)/100))]
    trainingPlus2SD = scores[int(97.5*(len(scores)/100))]
    trainingMinus1SD = scores[int(16*(len(scores)/100))]
    trainingMinus2SD = scores[int(2.5*(len(scores)/100))]

    if binary_leveled == 'leveled':
        if score < trainingMinus2SD:
            return "Extremely Negative"
        elif trainingMinus2SD <= score < trainingMinus1SD:
            return "Very Negative"
        elif trainingMinus1SD <= score < trainingMean:
            return "Somewhat Negative"
        elif trainingMean <= score < trainingPlus1SD:
            return "Somewhat Positive"
        elif trainingPlus1SD <= score < trainingPlus2SD:
            return "Very Positive"
        elif score > trainingPlus2SD:
            return "Extremely Positive"

    else:
        if score > trainingMean:
            return "Positive"
        else:
            return "Negative"

def main():
    #Main model run
    lazarCoversation1 = open('Lazar1.txt', 'r', encoding = "utf-8").readlines()
    lazarCoversation1 = list(map(lambda x: x.lower().strip("\n"), lazarCoversation1))
    testrun = runModel(lazarCoversation1)
    print(testrun[0])

    #Qualitative output
    sampleText = ["This is absolutely absurd I k it's not your fault but I'm never ordering from Domino's again. I've waited over three hours for my pizza and I'm so hungry and I'm crying"]
    qualitativeTest = runModel(sampleText)
    print(qualitativeOutput(qualitativeTest[0][0][0], binary_leveled = "leveled"))

    #Plotting functionality
    lazarCoversation1Scores = list(map(lambda x: x[0], testrun[0]))
    plotScores(lazarCoversation1Scores)

if __name__ == "__main__":
    main()
