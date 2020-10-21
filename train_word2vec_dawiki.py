from gensim.models.doc2vec import TaggedDocument
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import Doc2Vec
# class gensim.corpora.wikicorpus.WikiCorpus(fname
# <LANG>wiki-<YYYYMMDD>-pages-articles.xml.bz2


DATA_PATH = '../wiki/da/'
DAWIKI = 'dawiki-20201020-pages-articles.xml.bz2'

import os
import sys
# import bz2
import logging
import multiprocessing

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_PATH, 'model/')

DICTIONARY_FILEPATH = os.path.join(DATA_PATH, 'wiki-danish_wordids.txt.bz2')
WIKI_DUMP_FILEPATH = os.path.join(DATA_PATH, DAWIKI)


if __name__ == '__main__':

    # Check if the required files have been downloaded
    if not WIKI_DUMP_FILEPATH:
        print('Wikipedia articles dump could not be found..')
        print('Please see README.md for instructions!')
        sys.exit()


    # Get number of available cpus
    cores = multiprocessing.cpu_count()

    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # Initialize logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # if not os.path.isfile(DICTIONARY_FILEPATH):
    #     logging.info('Dictionary has not been created yet..')
    #     logging.info('Creating dictionary (takes about 9h)..')

    #     # Construct corpus
    #     wiki = WikiCorpus(WIKI_DUMP_FILEPATH)

    #     # Remove words occuring less than 20 times, and words occuring in more
    #     # than 10% of the documents. (keep_n is the vocabulary size)
    #     wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=100000)

    #     # Save dictionary to file
    #     wiki.dictionary.save_as_text(DICTIONARY_FILEPATH)
    #     del wiki

    # # Load dictionary from file
    # dictionary = gensim.corpora.Dictionary.load_from_text(DICTIONARY_FILEPATH)

    # Construct corpus using dictionary
    # wiki = gensim.corpora.WikiCorpus(WIKI_DUMP_FILEPATH, dictionary=dictionary)
    wiki = WikiCorpus(WIKI_DUMP_FILEPATH)

    class TaggedWikiDocumentIterator:
        def __init__(self, wiki):
            self.wiki = wiki
            self.wiki.metadata = True
        def __iter__(self):
            # for content, (page_id, title) in self.wiki.get_texts():
            for content, (_, title) in self.wiki.get_texts():
                yield TaggedDocument(content, [title])

    # class SentencesIterator:
    #     def __init__(self, wiki):
    #         self.wiki = wiki

    #     def __iter__(self):
    #         for sentence in self.wiki.get_texts():
    #             yield list(sentence)

    # Initialize simple sentence iterator required for the Word2Vec model
    # sentences = SentencesIterator(wiki)
    tagged_documents = TaggedWikiDocumentIterator(wiki=wiki)

    # logging.info('Training word2vec model..')
    # model = gensim.models.Word2Vec(sentences=, size=300, min_count=1, window=5, workers=cores)
    # model = gensim.models.Word2Vec(sentences=sentences, size=300, min_count=1, window=5, workers=cores)

    # models = [
    #     # PV-DBOW 
    #     Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #     # PV-DM w/average
    #     Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter =10, workers=cores),
    # ]

    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, min_count=17, iter=10, window=8, workers=cores)
    logging.info('Building vocabolary')
    model.build_vocab(tagged_documents)
    logging.info('Done building vocabolary')
    logging.info('Training doc2vec model')
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.iter)
    logging.info('Done training doc2vec model')

    # Save model
    logging.info('Saving model..')
    model.save(os.path.join(MODEL_PATH, 'dawiki_dbow_vs300_mc17_epochs10_window_8.model'))
    logging.info('Model saved')
    # logging.info('Done training word2vec model!')
