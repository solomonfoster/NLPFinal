import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sentimentTags = pd.read_csv('Sentiment_Tags.csv')
solomonTexts = pd.read_excel('SolomonTexts.xls')

solomonMegTexts = solomonTexts[solomonTexts['Address'] == '+16085153471']['Body'].tolist()
solomonMegTexts.reverse()

annaliseTexts = open('Annalise_Annalise.txt', 'r').readlines()
annaliseTexts = list(map(lambda x: x.lower().strip("\n"), annaliseTexts))

ryanTexts = open('Ryan_Ryan.txt', 'r').readlines()
ryanTexts = list(map(lambda x: x.lower().strip("\n"), ryanTexts))

sentimentWord = sentimentTags['Word'].tolist()
sentimentScore = sentimentTags['Score'].tolist()
sentimentDict = {}
for i in range(len(sentimentWord)):
    sentimentDict[sentimentWord[i]] = sentimentScore[i]

print(sentimentDict)
ax = plt.axes()
ax.set_title('Distribution of Scores for Tagged Words')
ax.set_ylabel('Score')
ax.set_xlabel('Words (Ordered)')
plt.scatter(list(range(len(sentimentDict.values()))), sentimentDict.values(), color = 'blue', marker = 'x', s = 15, label = 'score')
plt.show()

def computeSentenceAverages(inputText):
    cumulativeAverageScore, sentenceAverages = 0, []
    for i in range(len(inputText)):
        wordlist = inputText[i].lower().split(" ")
        cumulativeScore = 0
        for word in wordlist:
            word = word.strip('.').strip(',')
            try:
                score = sentimentDict[word]
            except KeyError:
                score = 4.5

            cumulativeScore += score

        sentenceAverageScore = cumulativeScore / len(wordlist)
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



if __name__ == "__main__":
    main()
