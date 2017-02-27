#
# Create Nietzsche dictionary
#
import nltk
from sklearn.externals import joblib

tags = {}
with open ( 'ntags.txt', 'r' ) as f:
    for line in f:
        sl = line.split()
        tags[ sl[ 0 ] ] = sl[ 1: ]

text = open('nietzsche.txt').read().lower()

text = text.decode("utf8")

tokens = nltk.word_tokenize( text )

pos = nltk.pos_tag( tokens )

for p in pos:
    if [ p[ 0 ] ] not in tags[ p[ 1 ] ]:
        tags[ p[ 1 ] ] += [ p[ 0 ] ]

joblib.dump( tags, 'niet.dict' )