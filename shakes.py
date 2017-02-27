# 
# Shakespeare HMM
# The Moore The Merrier
#
# Usage: python shakes.py [-t] [-g] [-d] modelname
#    -t        Train model
#    -g        Generate sonnet
#    -d        Use Shakespeare sonnet dictionary instead of Nietzsche
#    -a        Use all Shakespeare poems too for training and generation
#    -r        Follow rhymes when generating a sonnet
#    -m        Follow meter rules when generating a sonnet
#    modelname Save/load model parameters to/from modelname.pkl and modelname.posmn (must be last parameter)
#
import nltk
import hmmlearn
from hmmlearn import hmm
import random
import numpy as np
import sys
from sklearn.externals import joblib

## Helper functions

#
# Train the HMM
#
# Args: mn_sonnets - the array of sonnets x lines x multinomial-translated POS
#       n_pos      - number of different POS kinds in the sonnet
#
def train_hmm( mn_sonnets, n_pos ):
    #
    # Create model -- EDIT MODEL PARAMETERS HERE
    #
    n_iter = 1000
    n_components = n_pos
    model = hmm.MultinomialHMM( n_components=n_components, n_iter=n_iter, verbose=True )

    print "Training model with n_iter=%d, n_components=%d" % ( n_iter, n_components )
    # Prepare training data
    # Currently using each line independently
    X = []
    for sonnet in mn_sonnets:
        for line in sonnet:
            for token in line:
                if X == []:
                    X = [ token ]
                else:
                    X = np.concatenate( [ X, [ token ] ] )
    lengths = []
    for sonnet in mn_sonnets:
        for line in sonnet:
            lengths.append( len( line ) )

    model.fit( np.atleast_2d(X).T, lengths=lengths )

    return model


#
# Generate POS dictionary
#
# Args:    posmn             - list of POS types encountered
#          pos_sonnets       - the array for sonnets x lines x pos
# Returns: pos_dict          - POS dictionary of POS -> words in POS
def gen_posdict( posmn, pos_sonnets ):
    pos_dict = {}
    for item in posmn:
        pos_dict[ item ] = []

    for sonnet in pos_sonnets:
        for line in sonnet:
            for pos in line:
                pos_dict[ pos[ 1 ] ].append( pos[ 0 ] )

    return pos_dict

#
# Import and pre-process the data
#
# Args:    use_all    - True if should import all of Shakespeare, False for just sonnets
# Returns: mn_sonnets - the array of sonnets x lines x multinomial-translated POS
#          posmn      - the list of POS types encountered
#
def get_data( use_all ):
    # Import sonnets
    print "Importing sonnets..."
    sonnets = []
    with open( 'shakespeare.txt', 'r' ) as f:
        curr_sonnet = 0
        for line in f:
            # Ignore blank lines
            if line.replace(' ', '') == '\n':
                continue
            if line.strip() == str(curr_sonnet+1):
                if curr_sonnet != 0:
                    sonnets.append( lines )
                lines = []
                curr_sonnet += 1
            else:
                lines.append( line )
        sonnets.append( lines ) # last sonnet

    # Import others if necessary
    print "Importing everything else..."
    if use_all:
        with open( 'allshakes.txt', 'r' ) as f:
            lines = []
            for line in f:
                # Ignore blank lines
                if line.replace(' ', '') == '\n':
                    continue
                lines.append( line )
            sonnets.append( lines )

    print "Processing input..."
    # Tokenize the lines
    tokenized_sonnets = []
    for sonnet in sonnets:
        tokenized_sonnet = []
        for line in sonnet:
            tokenized_sonnet.append( nltk.word_tokenize( line ) )
        tokenized_sonnets.append( tokenized_sonnet )

    # Convert each token to a Part-Of-Speech
    pos_sonnets = []
    for sonnet in tokenized_sonnets:
        pos_sonnet = []
        for line in sonnet:
            pos_sonnet.append( nltk.pos_tag( line ) )
        pos_sonnets.append( pos_sonnet )

    # Now we have a bunch of parts of speech for our sonnets

    # Convert parts of speech to a multinomial distribution
    posmn = []
    for sonnet in pos_sonnets:
        for line in sonnet:
            for token in line:
                if token[ 1 ] not in posmn:
                    posmn.append( token[ 1 ] )

    mn_sonnets = []
    for sonnet in pos_sonnets:
        mn_sonnet = []
        for line in sonnet:
            mn_line = []
            for token in line:
                mn_line.append( posmn.index( token[ 1 ] ) )
            mn_sonnet.append( mn_line )
        mn_sonnets.append( mn_sonnet )

    return mn_sonnets, posmn, pos_sonnets

cmud = nltk.corpus.cmudict.dict()

#
# Count syllables in word (from stackexchange)
#
def nsyl(word):
  return [len(list(y for y in x if y[-1].isdigit())) for x in cmud[word.lower()]] 

#
# Check whether all words in a list rhyme
#
# Args:    words - list of words to be checked
# Returns: T/F   - whether all words rhyme
def do_they_rhyme( words ):
    determinant = []
    found = -1

    # Get the first word's syllables
    try:
        syllables = cmud[ words[ 0 ].lower() ][ 0 ]
    except KeyError:
        return False

    # Look for last stressed syllable in the word
    for i, syllable in enumerate( syllables ):
        if syllable[ -1 ] == '1' or syllable[ -1 ] == '2':
            found = i
    if found < 0:
        return False

    # And create a list of all the syllables after it
    for syllable in syllables[ found: ]:
        if syllable[ -1 ].isdigit():
            determinant.append( syllable[ :-1 ] )
        else:
            determinant.append( syllable )

    # Now check that all the other words' last few sillables match those of the first word
    for word in words[ 1: ]:
        try:
            for i, syllable in enumerate( cmud[ word.lower() ][ 0 ][ -( len( determinant ) ): ] ):
                if syllable[ -1 ].isdigit():
                    if syllable[ :-1 ] != determinant[ i ]:
                        return False
                else:
                    if syllable != determinant[ i ]:
                        return False
        except KeyError:
            return false
    return True


#
# Return stress for  a word as a list.
# 1 for stress, 0 for unstress, 2 for both, -1 for words not found in cmudict
#
def get_stress(word):
    stress = []
    punctuation = ["``", ":", "''", ".", "--", ","]

    if word.lower() in cmud.keys():
        for p in cmud[word.lower()]:
            curr_stress = []
            for i in range(len(p)):
                if p[i][-1].isdigit():
                    s = int(p[i][-1])
                    # do not differentiate secondary/primary stress
                    if s == 2:
                        curr_stress.append(1)
                    else:
                        curr_stress.append(s)
                        # count = len(stress)
            stress.append( curr_stress )

        final_stress = stress[ 0 ]
        for p in range( 1, len( stress ) ):
            if len( stress[ p ] ) != len( final_stress ):
                return [ -1 ]

            for i, syl in enumerate( stress[ p ] ):
                if final_stress[ i ] != 2 and syl != final_stress[ i ]:
                    final_stress[ i ] = 2
    else:
        return [ -1 ]

    return final_stress

## Runtime code

print "Using hmmlearn version " + hmmlearn.__version__
# Parse arguments
args = sys.argv[ 1: ]
files = [ a for a in args if a not in [ '-t', '-g', '-d', '-a', '-r', '-m' ] ]
if len( files ) > 0:
    modelfile = files[ -1 ]
else:
    modelfile = 'default'

# Train or import HMM parameters as required
if '-t' in args:
    if '-a' in args:
        mn_sonnets, posmn, _ = get_data( True )
    else:
        mn_sonnets, posmn, _ = get_data( False )
    model = train_hmm( mn_sonnets, len( posmn ) )
    joblib.dump( model, modelfile+'.pkl' )
    joblib.dump( posmn, modelfile+'.posmn' )
else:
    model = joblib.load( modelfile+'.pkl' )
    posmn = joblib.load( modelfile+'.posmn' )

# Generate or import POS dictionary
if '-d' in args:
    _, pmn, pos_sonnets = get_data( True )
    tags = gen_posdict( pmn, pos_sonnets )
else:
    tags = joblib.load( 'niet.dict' )
    # Import POS dictionary
    '''tags = {}
    with open ( 'tags.txt', 'r' ) as f:
        for line in f:
            sl = line.split()
            tags[ sl[ 0 ] ] = sl[ 1: ]'''

punctuation = [ "``", ":", "''", ".", "--", ",", "$", ")" ]


# Generate a sonnet
rhyme_pattern = [ 0, 1, 0, 1, 4, 5, 4, 5, 8, 9, 8, 9, 12, 12 ]
if '-g' in args:
    print "Generating sonnet:"

    rhymes = []

    # Generate each line independently
    for i in range( 14 ):
        # Find lines that satisfy all of our conditions
        done = False
        while not done:
            # Generate line
            emission = model.sample( 20 )
            etags = [ posmn[ e[ 0 ] ] for e in emission[ 0 ] ]
            if etags[ 0 ] in punctuation:
                continue
            # Process line
            words = []
            syllables = 0
            j = 0
            while True:
                candidates = tags[ etags[ j ] ]
                words.append( random.choice( candidates ) )
                if etags[ j ] not in punctuation:
                    if etags[ j ] == '(':
                        j += 1
                        continue
                    try:
                        syllables += nsyl( words[ j ] )[ 0 ]
                    except KeyError:
                        break
                    if syllables > 10:
                        break
                else:
                    if syllables == 10:
                        done = True
                        break
                j += 1

            # Check meter
            if '-m' in args:
                if done:
                    stress = []
                    for j, w in enumerate( words ):
                        if etags[ j ] not in punctuation and w != "'":
                            stress.append( get_stress( w ) )
                    index = 0
                    for w in stress:
                        for s in w:
                            if s != 2 and s != ( index % 2 ):
                                done = False
                                break
                            index += 1

            # Check rhyme -- THIS HAS TO BE DONE LAST
            if '-r' in args:
                if done:
                    try:
                        # First check that word is in our dictionary
                        test = cmud[ words[ -2 ].lower() ]

                        # Now check rhyme
                        if rhyme_pattern[ i ] == i:
                            rhymes.append( words[ -2 ] )
                        else:
                            if not do_they_rhyme( [ words[ -2 ], rhymes[ rhyme_pattern[ i ] ] ] ):
                                done = False
                            else:
                                rhymes.append( words[ -2 ] )
                    except KeyError:
                        done = False

        text = ''
        for j, word in enumerate( words ):
            if etags[ j ] not in punctuation and j != 0:
                text += ' '
            # Fix punctuation on last word
            good_punctuation = [ ',', '.', '!', '?', '...', ';' ]
            if j == len( words ) - 1:
                if i == 13:
                    text += '.'
                elif word not in good_punctuation:
                    better_punct = random.choice( good_punctuation )
                    text += better_punct
                    last_punctuation = better_punct
                else:
                    last_punctuation = word
                    text += word
            # Fix capitalization on first word
            elif j == 0 and ( i == 0 or last_punctuation in [ '.', '!', '?', '...' ] ):
                text += ( word[ 0 ].upper() + word[ 1: ] )
            # Just print other words
            else:
                text += word.lower()

        # Print the generated sonnet

        print text
