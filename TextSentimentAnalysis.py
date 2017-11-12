import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
import string
import re

stop_words = text.ENGLISH_STOP_WORDS

sentimentTags = pd.read_csv('Sentiment_Tags.csv')
#solomonTexts = pd.read_excel('SolomonTexts.xls')
#
#solomonMegTexts = solomonTexts[solomonTexts['Address'] == '+16085153471']['Body'].tolist()
#solomonMegTexts.reverse()

annaliseTexts = open('Annalise_Annalise.txt', 'r', encoding = "utf-8").readlines()
annaliseTexts = list(map(lambda x: x.lower().strip("\n"), annaliseTexts))

#ryanTexts = open('Ryan_Ryan.txt', 'r', encoding = 'utf-8').readlines()
#ryanTexts = list(map(lambda x: x.lower().strip("\n"), ryanTexts))

sentimentWord = sentimentTags['Word'].tolist()
sentimentScore = sentimentTags['Score'].tolist()
sentimentDict = {}
for i in range(len(sentimentWord)):
    sentimentDict[sentimentWord[i]] = sentimentScore[i]

#ax = plt.axes()
#ax.set_title('Distribution of Scores for Tagged Words')
#ax.set_ylabel('Score')
#ax.set_xlabel('Words (Ordered)')
#plt.scatter(list(range(len(sentimentDict.values()))), sentimentDict.values(), color = 'blue', marker = 'x', s = 15, label = 'score')
#plt.show()

def computeSentenceAverages(inputText):
    cumulativeAverageScore, sentenceAverages = 0, []
    for i in range(len(inputText)):
        wordlist = inputText[i].lower().split(" ")
        cumulativeScore = 0
        stopWordPunctScore = 0
        for word in wordlist:
            word =  "".join(l for l in word if l not in string.punctuation)
            if word in stop_words or word in string.punctuation:
                stopWordPunctScore = stopWordPunctScore + 1
                continue
            else:
                try:
                    score = sentimentDict[word]
                except KeyError:
                    score = 4.5

            cumulativeScore += score

        sentLenStopWord = len(wordlist) - stopWordPunctScore
        if sentLenStopWord == 0:
            sentenceAverageScore = 4.5
        else:
            sentenceAverageScore = cumulativeScore / sentLenStopWord
        print("Score:", cumulativeScore)
        print("Length:", len(wordlist))
        print("StopWords:", stopWordPunctScore)
        print("Final:", sentenceAverageScore)
        cumulativeAverageScore += sentenceAverageScore
        sentenceAverages.append(sentenceAverageScore)

    return [sentenceAverages, inputText]

def bigramModel():
    solomonTextData = solomonTexts['Body'].tolist()
    bigramDict = {}

    return bigramDict

def plotScores(textdata):
    ax = plt.axes()
    ax.set_ylabel("Score")
    ax.set_title("Solomon Sent Texts")
    plt.scatter(list(range(len(textdata))), textdata, color = 'blue', marker = 'o', label = 'score')
    plt.show()

def main():
    # bigramModel()
    annaliseAvgs, annaliseText_ = computeSentenceAverages(annaliseTexts)
    combinedList = []
    for i in range(len(annaliseAvgs)):
        combinedList.append([annaliseAvgs[i], annaliseText_[i]])

    combinedList.sort()
    
    print(combinedList)



if __name__ == "__main__":
    main()
