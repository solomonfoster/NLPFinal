import pandas as pd
from nltk.metrics.distance import edit_distance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
import string
from nltk.corpus import wordnet
from gensim.models import Word2Vec

stop_words = text.ENGLISH_STOP_WORDS

sentimentTags = pd.read_csv('Sentiment_Tags.csv')
solomonTexts = pd.read_excel('SolomonTexts.xls')
#
solomonMegTexts = solomonTexts[solomonTexts['Address'] == '+16085153471']['Body'].tolist()
solomonMegTexts.reverse()

annaliseTexts = open('Annalise_Annalise.txt', 'r', encoding = "utf-8").readlines()
annaliseTexts = list(map(lambda x: x.lower().strip("\n"), annaliseTexts))

#ryanTexts = open('Ryan_Ryan.txt', 'r', encoding = 'utf-8').readlines()
#ryanTexts = list(map(lambda x: x.lower().strip("\n"), ryanTexts))

sentimentWord = sentimentTags['Word'].tolist()
sentimentScore = sentimentTags['Score'].tolist()
sentimentDict = {}
for i in range(len(sentimentWord)):
    sentimentDict[sentimentWord[i]] = sentimentScore[i]

# def word2vecModelBuild(sentences):
#    model = Word2Vec(sentences)
#    return model
#
# def word2vecSmooth(model, word):
#    closeWord = model.most_similar(word)[0][0]
#    print(closeWord)
#    score = sentimentDict[closeWord]
#    return score

def plotScores(textdata):
    ax = plt.axes()
    ax.set_ylabel("Score")
    ax.set_title("Solomon Sent Texts")
    plt.scatter(list(range(len(textdata))), textdata, color = 'blue', marker = 'o', label = 'score')
    plt.show()

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
def handleMisspellings(word):
    distance = 10
    for i, referenceWord in enumerate(sentimentWord):
        editDistance = edit_distance(word, referenceWord)
        if editDistance < distance:
            closestWord, closestWordIndex = referenceWord, i
            distance = editDistance

    print(closestWord)
    # return score

#Helper function for computeSentenceAverages: tries to find synonyms for words not in tagset
def wordNetSmooth(word):
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
        score = "<unk>"
    return score

def computeSentenceAverages(inputText):
    cumulativeAverageScore, sentenceAverages = 0, []
    sentenceList = cleanInputData(inputText)
    individualWordScores = []
    for sent in range(len(sentenceList)):
        cumulativeScore, stopWordPunctScore, unknownCount = 0, 0, 0
        wordlist = sentenceList[sent]
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
                    score = wordNetSmooth(word)

            if score == "<unk>":
                unknownCount += 1
            else:
                cumulativeScore += score

        #Do not count stop words or punctuation toward average
        sentLenStopWord = len(wordlist) - stopWordPunctScore - unknownCount
        if sentLenStopWord == 0:
            sentenceAverageScore = 4.5
        else:
            sentenceAverageScore = cumulativeScore / sentLenStopWord

        cumulativeAverageScore += sentenceAverageScore
        sentenceAverages.append(sentenceAverageScore)

    return [sentenceAverages, inputText]

def runModel(text):
    textAvgs, textText = computeSentenceAverages(text)
    combinedList = []
    percentSum = 0
    for i in range(len(textAvgs)):
        combinedList.append([textAvgs[i], textText[i]])

    combinedList.sort()
    print(combinedList)


def main():
    sampleSentences = ["i'm too drunk and too high"]
    # runModel(sampleSentences)
    runModel(sampleSentences)
    handleMisspellings("ahahahaha")

if __name__ == "__main__":
    main()
