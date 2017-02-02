import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
from collections import Counter, defaultdict, deque
import copy
import math
import urllib.request
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


# download sentiment list and store them into a dict
def read_sentiment_list():
	url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
	zipfile = ZipFile(BytesIO(url.read()))
	afinn_file = zipfile.open('AFINN/AFINN-111.txt')

	afinn = dict()

	for line in afinn_file:
		parts = line.strip().split()
		if len(parts) == 2:
			afinn[parts[0].decode("utf-8")] = int(parts[1])

	# print('read %d AFINN terms.\nE.g.: %s' % (len(afinn), str(list(afinn.items())[:10])))

	return afinn

# rate message score
def afinn_sentiment(terms, afinn):
    total = 0
    for t in terms:
        if t in afinn:
            # print('\t%s=%d' % (t, afinn[t]))
            total += afinn[t]
    return total
"""
read output_message.txt and rate each tweet, then output sentiment 
result into output_emotion.txt
"""
def rate_tweets(filename, afinn):
    total = 0
    good = 0
    neutral = 0
    bad = 0
    text_file1 = open("output_good_emotion.txt", "w")
    text_file2 = open("output_neutral_emotion.txt", "w")
    text_file3 = open("output_bad_emotion.txt", "w")
    text_file4 = open("output_summary_emotion.txt", "w")

    for line in open(filename):
        l = line.rstrip('\n')
        if len(l) > 0:
            score = afinn_sentiment(line.split(), afinn)
            total += 1
            if score > 0:
                good += 1
                text_file1.write('%s = %d\n' % (line, score))
            elif score < 0:
                bad += 1
                text_file3.write('%s = %d\n' % (line, score))
            else:
                neutral += 1
                text_file2.write('%s = %d\n' % (line, score))
    text_file4.write('Emotion Summary of Tweets\n')
    text_file4.write('Total number of tweets: %d\n' % total)
    text_file4.write('Good emotion: %d\n' % good)
    text_file4.write('Neutral emotion: %d\n' % neutral)
    text_file4.write('Bad emotion: %d' % bad)
    text_file1.close()
    text_file2.close()
    text_file3.close()
    text_file4.close()


def main():
	afinn = read_sentiment_list()
	rate_tweets('output_message.txt', afinn)


if __name__ == '__main__':
    main()