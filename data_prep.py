import os
import re
import sys
import re
import spacy
import io
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple, Counter
from itertools import combinations
from multiprocessing import cpu_count
from sklearn.model_selection import train_test_split

# tokenizer
nlp = spacy.load('en')


# Special vocabulary symbols.
PAD_TOKEN = '<pad>' # pad symbol
UNK_TOKEN = '<unk>' # unknown word
BOS_TOKEN = '<bos>' # begin-of-sentence symbol
EOS_TOKEN = '<eos>' # end-of-sentence symbol
NUM_TOKEN = '<num>' # numbers

# we always put them at the start.
_START_VOCAB = [PAD_TOKEN, UNK_TOKEN]
PAD_ID = 0
UNK_ID = 1

vocab_ints = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
    BOS_TOKEN: 2,
    EOS_TOKEN: 3,
    NUM_TOKEN: 4
}

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile(r"^\d+$")


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
RANDOM_SEED = 1234


# NER tags to split on
# https://spacy.io/api/annotation#named-entities
tag_map = dict(
    ORG='',
    GPE='',
    PERSON='',
)

sentence_tuple = namedtuple('Sentence', 'doc_id sent_id left middle right')


def basic_tokenizer(sequence, bos=True, eos=True):
    sequence = re.sub(r'\s{2}', ' ' + EOS_TOKEN + ' ' + BOS_TOKEN + ' ', sequence)
    if bos:
        sequence = BOS_TOKEN + ' ' + sequence.strip()
    if eos:
        sequence = sequence.strip() + ' ' + EOS_TOKEN
    return sequence.lower().split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=40000, tokenizer=None, bos=True, eos=True):
    """Create vocabulary file (if it does not exist yet) from data file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(vocabulary_path):
        print("Creating vocabulary {} from data {}".format(vocabulary_path, data_path))
        vocab = Counter()
        with io.open(data_path, mode="r", encoding='utf-8') as f:
            for line in f.readlines():
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line, bos, eos)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, NUM_TOKEN, w)
                    vocab[word] += 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                print(" \t{:,} words found. Truncate to {:,}.".format(len(vocab_list), max_vocabulary_size))
                vocab_list = vocab_list[:max_vocabulary_size]
            with io.open(vocabulary_path, mode="w", encoding="utf-8") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path, binary=False):
    """Initialize vocabulary from file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if os.path.exists(vocabulary_path):
        rev_vocab = []
        if binary is True:
            with io.open(vocabulary_path, mode='rb') as f:
                rev_vocab.extend(f.readlines())
        else:
            with io.open(vocabulary_path, mode='r', encoding='utf-8') as f:
                rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file {} not found.".format(vocabulary_path))


def prepare_ids(data_dir, vocab_path):
    for context in ['left', 'middle', 'right', 'combined']:
        for _set in ['train', 'test']:
            data_path = os.path.join(data_dir, _set, 'source.{}.txt'.format(context))
            target_path = os.path.join(data_dir, _set, 'ids.{}.txt'.format(context))
            if context == 'left':
                bos, eos = True, False
            elif context == 'middle':
                bos, eos = False, False
            elif context == 'right':
                bos, eos = False, True
            else:
                bos, eos = True, True
            data_to_token_ids(data_path, target_path, vocab_path, bos=bos, eos=eos)


def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer=None, bos=True, eos=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    if not os.path.exists(target_path):
        print("Vectorizing data in {}".format(data_path))
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with io.open(data_path, mode="r", encoding="utf-8") as data_file:
            with io.open(target_path, mode="w", encoding="utf-8") as tokens_file:
                for line in data_file:
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer, bos, eos)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, bos=True, eos=True):
    """Convert a string to list of integers representing token-ids.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    words = tokenizer(sentence) if tokenizer else basic_tokenizer(sentence, bos, eos)
    return [vocabulary.get(re.sub(_DIGIT_RE, NUM_TOKEN, w), UNK_ID) for w in words]


def normalize(text, tokenizer=None, bos=True, eos=True):
    """Convert a string to list of integers representing token-ids.

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/data_utils.py
    """
    output = []
    tokens = tokenizer(text) if tokenizer else basic_tokenizer(text, bos, eos)
    for w in tokens:
        output.append(get_vocab_int(w))
    return ' '.join(output)


def get_vocab_int(text):
    """
    Build vocab dynamically

    :param text:
    :return:
    """
    word = re.sub(_DIGIT_RE, NUM_TOKEN, text)
    vocab_ints[word] = vocab_ints.get(word, len(vocab_ints) + 1)
    return vocab_ints.get(word)


def get_noun_chunks(parsed):
    nc = defaultdict(list)
    for chunk in parsed.noun_chunks:
        nc[chunk.sent.start_char].append(dict(
            begin=chunk.start_char,
            end=chunk.end_char,
            text=chunk.text,
            tag=chunk.root.dep_
        ))

    return nc

def get_named_entities(parsed):
    ner = defaultdict(list)
    for ent in parsed.ents:
        if ent.label_ not in tag_map.keys():
            continue

        ner[ent.sent.start_char].append(dict(
            begin=ent.start_char,
            end=ent.end_char,
            text=ent.text,
            tag=ent.label_
        ))
    return ner


def spacy_tokenizer(doc):
    # return ' '.join(doc.split())
    return ' '.join(re.sub('([^0-9a-zA-Z\'])', ' \\1 ', doc).split())
    # output = []
    # for token in doc:
    #     if token.is_space or token.is_punct:
    #         continue
    #     elif token.is_numeric:
    #         output.append(NUM_TOKEN)
    #     else:
    #         output.append(token.lemma_)
    #
    # return ' '.join(output)


def get_sentence_combinations(doc_id, splits, parsed, tokenizer=None):
    if tokenizer is None:
        tokenizer = spacy_tokenizer

    rows = []

    for sent_id, sentence in enumerate(parsed.sents):
        # get named entities in this sentence
        # TODO: replace NER partitioning with noun_chunks?
        sent_splits = splits.get(sentence.start_char, [])
        sent = None
        if len(sent_splits) == 0 :
            # no entities to split on, likely improper sentence
            # TODO: place full sentence in left or split evenly on tokens?
            sent = [sentence_tuple(
                doc_id=doc_id,
                sent_id=sent_id,
                left=tokenizer(sentence.string),
                middle='',
                right=''
            )]
        elif len(sent_splits) == 1:
            # only one entity to split on, neglect middle
            # simulate call where two named entities are immediately next to one another
            left_entity = sent_splits[0]['text']
            left_entity_begin = sent_splits[0]['begin'] - sentence.start_char
            left_entity_end = sent_splits[0]['end'] - sentence.start_char
            left_entity_tag = sent_splits[0]['tag']

            s = sentence.string
            left = s[:left_entity_begin] + left_entity
            right = s[left_entity_end:]

            sent = [sentence_tuple(
                doc_id=doc_id,
                sent_id=sent_id,
                left=tokenizer(left),
                middle='',
                right=tokenizer(right)
            )]
        else:
            # if there are more than two, process each combination
            sent = []
            for c in combinations(sent_splits, 2):
                # TODO: should we skip when two NER are immediately next to one another?
                # TODO: the original implementation does not skip...

                left_entity = c[0]['text']
                left_entity_begin = c[0]['begin'] - sentence.start_char
                left_entity_end = c[0]['end'] - sentence.start_char
                left_entity_tag = c[0]['tag']

                right_entity = c[1]['text']
                right_entity_begin = c[1]['begin'] - sentence.start_char
                right_entity_end = c[1]['end'] - sentence.start_char
                right_entity_tag = c[1]['tag']

                s = sentence.string
                left = s[:left_entity_begin] + left_entity
                right = right_entity + s[right_entity_end:]
                middle = s[left_entity_end:right_entity_begin]
                sent.append(sentence_tuple(
                    doc_id=doc_id,
                    sent_id=sent_id,
                    left=tokenizer(left),
                    middle=tokenizer(middle),
                    right=tokenizer(right)
                ))

        if sent is not None:
            rows.extend(sent)

    return rows


def split_sentence_to_vocab_ints(source_filename, data_column='clean_text', is_train=True):
    """read NER output files and store them in a pandas DataFrame"""
    _X = []

    # explicitly load observations with sequential index
    df_raw = pd.read_csv(source_filename).fillna('')

    # specify the train/test split for multi-instance
    df_raw['is_test'] = False
    if is_train is True:
        # apply basic huristic on splitting based on the number of times a given label appears first
        # TODO: uncertain this is providing a balanced split
        df_raw['first_label'] = df_raw['label'].apply(lambda x: x.split(',')[0])
        x_train, x_test = train_test_split(
            df_raw,
            test_size=.25,
            shuffle=True,
            stratify=df_raw['first_label']
        )
        df_raw.loc[x_train.index, ['is_test']] = True
        df_raw = df_raw.drop('first_label', axis=1)

    # get sentence combinations for each observations, in order but multithreaded
    # https://spacy.io/usage/processing-pipelines#section-multithreading
    for doc_id, parsed in enumerate(nlp.pipe(
            texts=df_raw[data_column],
            batch_size=5000,
            n_threads=cpu_count() - 1
    )):
        # splits = get_named_entities(parsed)
        splits = get_noun_chunks(parsed)
        combos = get_sentence_combinations(doc_id, splits, parsed)
        _X.extend(combos)

    # join back with input, for label allocation to proper documents/sentences
    df_out = pd.DataFrame(_X, columns=_X[-1]._fields).merge(
        df_raw.reset_index().drop(data_column, axis=1),
        left_on='doc_id',
        right_on='index'
    ).drop('index', axis=1)

    return df_out


def _df_to_disk(df, column, path, force=False):
    dst = os.path.join(path, 'source.{}.txt'.format(column))
    if not force and os.path.isfile(dst):
        print('File already exists: {}'.format(dst))
        return

    return _df_write_to_disk(dst, df[column])


def _df_write_to_disk(dst, df):
    with io.open(dst, mode='w', encoding='utf-8') as f:
        for row in df.tolist():
            f.write(row + '\n')


def preserve_sentence_to_disk(df, path, label_column='label', force=True):
    idx = df['is_test'] == True
    df = df.fillna('')
    df['combined'] = df['left'] + ' ' + df['middle'] + ' ' + df['right']

    # identify distinct labels
    labels = set()
    for x in df[label_column].unique():
        for y in x.split(','):
            labels.add(y.strip())

    labels = sorted(labels)
    binary_ize = lambda x: ' '.join(['1' if y in list(map(str.strip, x.split(','))) else '0' for y in labels])

    # preserve train cases, if any
    if idx.sum() > 0:
        train = os.path.join(path, 'train')
        if not os.path.isdir(train):
            os.makedirs(train)
        _df_to_disk(df[idx], 'left', train, force=force)
        _df_to_disk(df[idx], 'right', train, force=force)
        _df_to_disk(df[idx], 'middle', train, force=force)
        _df_to_disk(df[idx], 'combined', train, force=force)
        _df_write_to_disk(os.path.join(train, 'targets.txt'), df[idx][label_column].apply(binary_ize))


    # preseve test cases
    test = os.path.join(path, 'test')
    if not os.path.isdir(test):
        os.makedirs(test)
    _df_to_disk(df[~idx], 'left', test, force=force)
    _df_to_disk(df[~idx], 'right', test, force=force)
    _df_to_disk(df[~idx], 'middle', test, force=force)
    _df_to_disk(df[~idx], 'combined', test, force=force)
    _df_write_to_disk(os.path.join(test, 'targets.txt'), df[~idx][label_column].apply(binary_ize))


def _load_bin_vec(fname, vocab_path):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec

    Original taken from
    https://github.com/yuhaozhang/sentence-convnet/blob/master/text_input.py
    """
    vocab, _ = initialize_vocabulary(vocab_path, binary=True)
    word_vecs = dict()
    with io.open(fname, mode="rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            if word in vocab:
                # read in vector from file into memory
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                # skip over word from vector file, it's not recognized from the input text
                f.read(binary_len)
    return word_vecs, layer1_size, vocab


def _load_txt_vec(fname, vocab_path):
    vocab, _ = initialize_vocabulary(vocab_path, binary=False)
    word_vecs = dict()
    layer1_size = None
    with io.open(fname, mode="r", encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if layer1_size is None:
                layer1_size = len(parts) - 1
            if parts[0] in vocab:
                word_vecs[parts[0]] = np.array(parts[1:], dtype='float32')
    return word_vecs, layer1_size, vocab
    


def _add_random_vec(word_vecs, vocab, emb_size=300):
    """
    For any vectors not in our corpus, add a random vector so we have some representation

    :param word_vecs:
    :param vocab:
    :param emb_size:
    :return:
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,emb_size)
    return word_vecs


def prepare_pretrained_embedding(fname, vocab_path):
    print('Reading pretrained word vectors from file ...')
    if fname.endswith('bin'):
        word_vecs, emb_size, word2id = _load_bin_vec(fname, vocab_path)
    else:
        word_vecs, emb_size, word2id = _load_txt_vec(fname, vocab_path)
    word_vecs = _add_random_vec(word_vecs, word2id, emb_size)
    embedding = np.zeros([len(word2id), emb_size])
    for w, idx in word2id.items():
        embedding[idx,:] = word_vecs[w]
    print('Generated embeddings with shape ' + str(embedding.shape))
    return embedding


def create_source_file(src_path, data_path, data_column=''):
    with io.open(src_path, mode='w', encoding='utf-8') as f:
        for row in pd.read_csv(data_path)[data_column].tolist():
            f.write(row + '\n')


def extract_text_from_file(dst_path, data_path, data_column):
    with io.open(dst_path, mode='w', encoding='utf-8') as f:
        for line in pd.read_csv(data_path)[data_column].fillna('').tolist():
            f.write(line + '\n')


def __main(input_filename, data_column='clean_text', max_vocab_size=36500, embedding_filename='word2vec/GoogleNews-vectors-negative300.bin', is_train=False):
    THIS_DIR = os.path.dirname(__file__)
    # data_dir = os.path.join(THIS_DIR, 'data')
    dest_dir = os.path.join(THIS_DIR, 'data', os.path.basename(os.path.splitext(input_filename)[0]))
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    # condense text into a flat-file for use with vocab generation
    src_path = os.path.join(dest_dir, 'vocab-source.txt')
    if not os.path.isfile(src_path):
        extract_text_from_file(src_path, input_filename, data_column=data_column)

    # process vocab into int
    vocab_path = os.path.join(dest_dir, 'vocab.txt')
    if not os.path.isfile(vocab_path):
        create_vocabulary(vocab_path, src_path, max_vocab_size)

    # multi-label multi-instance (MLMI-CNN) dataset
    df_path = os.path.join(dest_dir, 'sentence_splits.csv')
    if not os.path.isfile(df_path):
        df = split_sentence_to_vocab_ints(input_filename, data_column=data_column, is_train=is_train)
        df.to_csv(df_path, index=False)
    else:
        df = pd.read_csv(df_path)

    # preserve sentence to disk
    preserve_sentence_to_disk(df, dest_dir)

    # convert text into index position of vocab
    prepare_ids(dest_dir, vocab_path)

    # pretrained embeddings
    embedding_path = os.path.join(THIS_DIR, embedding_filename)
    emb_path = os.path.join(dest_dir, 'emb.npy')
    if not os.path.isfile(emb_path):
        embedding = prepare_pretrained_embedding(embedding_path, vocab_path)
        np.save(emb_path, embedding)


def read_data_contextwise(source_path, sent_len):
    """Read source file and pad the sequence to sent_len,
       combine them with target (and attention if given).

    Original taken from
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/translate/translate.py
    """
    print("Loading data...")
    data = dict(
        train=None,
        test=None
    )
    for _set in data.keys():
        _X = {'left': [], 'middle': [], 'right': []}
        for context in _X.keys():
            path = os.path.join(source_path, _set, 'ids.{}.txt'.format(context))
            with io.open(path, mode="r", encoding="utf-8") as source_file:
                for source in source_file.readlines():
                    source_ids = [np.int64(x) for x in source.split()]
                    if sent_len > len(source_ids):
                        source_ids += [np.int64(PAD_ID)] * (sent_len - len(source_ids))
                    assert len(source_ids) == sent_len
                    _X[context].append(source_ids)

        assert len(_X['left']) == len(_X['middle'])
        assert len(_X['right']) == len(_X['middle'])

        _y = []
        path = os.path.join(source_path, _set, 'targets.txt')
        with io.open(path, mode="r", encoding="utf-8") as target_file:
            for target in target_file.readlines():
                target_ids = [np.float32(y) for y in target.split()]
                _y.append(target_ids)

        _y = np.array(_y)

        assert len(_X['left']) == len(_y)
        print("\t{:,} {} examples found.".format(len(_y), _set))

        _at = None
        path = os.path.join(source_path, _set, 'attention.txt')
        if os.path.isfile(path):
            with io.open(path, mode="r", encoding="utf-8") as att_file:
                _at = [np.float32(att) for att in att_file.readlines()]
                assert len(_at) == len(_y)

        _a = np.array([None] * _y.shape[0])
        if _at is not None and len(_at) == len(_y):
            _a = np.array(_at)
            # compute softmax
            _a = np.reshape(np.exp(_a) / np.sum(np.exp(_a)), (_y.shape[0], 1))
            assert _a.shape[0] == _y.shape[0]

        data[_set] = np.array(list(zip(
            np.array(_X['left']),
            np.array(_X['middle']),
            np.array(_X['right']),
            _y,
            _a
        )))

    return data['train'], data['test']


if __name__ == '__main__':
    input_filename = os.path.abspath(sys.argv[1])
    max_vocab_size = 36500
    embedding_filename = 'word2vec/glove.42B.300d.txt'
    __main(input_filename, max_vocab_size=max_vocab_size, embedding_filename=embedding_filename, is_train=True)
