from gensim.models.doc2vec import TaggedDocument
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import Doc2Vec

DATA_PATH = '../wiki/da/'
DAWIKI = 'dawiki-20201020-pages-articles.xml.bz2'

import os
import sys
import logging
import multiprocessing

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_PATH, 'model/')

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

    wiki = WikiCorpus(WIKI_DUMP_FILEPATH)

    class TaggedWikiDocumentIterator:
        def __init__(self, wiki):
            self.wiki = wiki
            self.wiki.metadata = True
        def __iter__(self):
            # for content, (page_id, title) in self.wiki.get_texts():
            for content, (_, title) in self.wiki.get_texts():
                yield TaggedDocument(content, [title])


    # Initialize TaggedWikiDocument iterator for the Doc2Vec model
    tagged_documents = TaggedWikiDocumentIterator(wiki=wiki)

    # Example model from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-wikipedia.ipynb
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
