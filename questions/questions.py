import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = {}
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding='utf8') as f:
            contents[file] = f.read()
    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # nltk.download('stopwords')
    return [
        word for word in nltk.word_tokenize(document.lower())
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words('english')
    ]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # inverse document frequency of a word is defined by taking the natural logarithm of the number of documents divided by the number of documents in which the word appears
    idf = {}

    # Concatenating dictionary value lists
    # Using sum() + values()
    words = set(sum(documents.values(), []))

    for word in words:
        doc_cnt = 0
        for file in documents:
            if word in documents[file]:
                doc_cnt += 1
        idf[word] = math.log(len(documents) / doc_cnt)
    return idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidf = {}    
    for file in files:
        score = 0
        for q in query:
            tf = 0
            for word in files[file]:
                if q == word:
                    tf += 1
            score += tf * idfs[q]
        if score != 0:
            tfidf[file] = score
    # sort in desscending order
    tfidf_sorted = [k for k, v in sorted(tfidf.items(), key=lambda x:x[1], reverse=True)]
    return tfidf_sorted[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    score = {}
    for sentence in sentences:
        idf = 0
        occur = 0
        for q in set(query):
            if q in sentences[sentence]:
                idf += idfs[q]
                occur += 1
        if idf != 0 and occur != 0:
            score[sentence] = (idf, (occur / len(sentence)))

    score_sorted = [k for k, v in sorted(score.items(), key=lambda x:(x[1][0], x[1][1]), reverse=True)]
    return score_sorted[:n]


if __name__ == "__main__":
    main()
