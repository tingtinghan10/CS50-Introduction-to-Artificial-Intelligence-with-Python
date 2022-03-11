import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP
NP -> N | N AP | N VP | N CP | Det NP | N PP | N AdvP
VP -> V | V AP | V NP | V PP | V AdvP
PP -> P AP | P NP
AP -> Adj AP | Det AP | Adj CP | Adj NP | Adj VP
CP -> Conj NP | Conj VP
AdvP -> Adv | Adv CP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = []
    for word in nltk.word_tokenize(sentence):
        if word.isalpha():
            words.append(word.lower())
    return words


def has_np(tree):
    """
    returns True is there is NP in the subtrees and False if there isn't
    """
    # terminal node has height 2
    if tree.height() == 2:
        return tree.label() == 'NP'

    # check all subtrees excluding current tree
    for subtree in tree.subtrees(lambda t:t != tree):
        if subtree.label() == 'NP':
            return True
        if has_np(subtree):
            return True
    
    return False


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    # '.label()' returns node label of the tree
    # '.subtrees(filter=None)' generates all the subtrees of this tree (inclusive of current tree, aka the S tree), optionally restricted to trees matching the filter function.
    # tree.subtrees() -> returns also the current tree
    # for subtree in tree: -> returns only the two branches of the current tree
    np = []
    for subtree in tree.subtrees():
        if subtree.label() == 'NP' and has_np(subtree) == False:
            np.append(subtree)
    return np


if __name__ == "__main__":
    main()
