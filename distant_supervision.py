# -*- coding: utf-8 -*-

##########################################################
#
# Distant Supervision
#
###########################################################

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs

import os
import re
import collections
import time
import requests
from urllib.request import urlopen
import glob
import io
from itertools import combinations
from collections import Counter
import util

import spacy



# global variables
nlp = spacy.load('en')
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
orig_dir = os.path.join(data_dir, 'orig')
ner_dir = os.path.join(data_dir, 'ner')


ner_path = "/usr/local/Cellar/stanford-ner/3.5.2/libexec/"
stanford_classifier = os.path.join(ner_path, 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
stanford_ner = os.path.join(ner_path, 'stanford-ner.jar')

# tag_map = {
#     'ORGANIZATION': 'Q43229',  # https://www.wikidata.org/wiki/Q43229
#     'LOCATION': 'Q17334923',   # https://www.wikidata.org/wiki/Q17334923
#     'PERSON': 'Q5'             # https://www.wikidata.org/wiki/Q5
# }

# https://spacy.io/api/annotation#named-entities
tag_map = dict(
    ORG='Q43229',
    GPE='Q17334923',
    PERSON='Q5',
)

# column names in DataFrame
col = [
    'doc_id', 'sent_id', 'sent', 'subj', 'subj_begin', 'subj_end', 'subj_tag',
    'rel', 'obj', 'obj_begin', 'obj_end', 'obj_tag'
]


def sanitize(string):
    """clean wikipedia article"""
    string = re.sub(r"\[\d{1,3}\]", " ", string)
    string = re.sub(r"\[edit\]", " ", string)
    string = re.sub(r" {2,}", " ", string)
    return string.strip()


def download_wiki_articles(doc_id, limit=100, retry=False):
    """download wikipedia article via Mediawiki API"""
    base_path = "http://en.wikipedia.org/w/api.php?format=xml&action=query"
    query = base_path + "&list=random&rnnamespace=0&rnlimit=%d" % limit
    r = None
    try:
        r = urlopen(query).read()
    except Exception as e:
        if not retry:
            download_wiki_articles(doc_id, limit, retry=True)
        else:
            print(e)
            return None
    pages = bs(r, "html.parser").findAll('page')
    if len(pages) < 1:
        return None
    docs = []
    for page in pages:
        if int(page['id']) in doc_id:
            continue

        link = base_path + "&prop=revisions&pageids=%s&rvprop=content&rvparse" % page['id']
        content = urlopen(link).read()
        content = bs(content, "html.parser").find('rev').stripped_strings

        # extract paragraph elements only
        text = ''
        for p in bs(' '.join(content), "html.parser").findAll('p'):
            text += ' '.join(p.stripped_strings) + '\n'
        #text = text.encode('utf8')
        text = sanitize(text)

        # save
        if len(text) > 0:
            title = re.sub(r"[ /]", "_", page['title'])
            filename = page['id'] + '-' + title + '.txt'
            docs.append(filename)
            with io.open(os.path.join(orig_dir, filename), mode='w', encoding='utf-8') as f:
                f.write(text)
    return docs


def exec_ner(filenames):
    """execute Stanford NER"""
    for filename in filenames:
        in_path = os.path.join(orig_dir, filename)
        out_path = os.path.join(ner_dir, filename)
        cmd = 'java -mx700m -cp "%s:" edu.stanford.nlp.ie.crf.CRFClassifier' % stanford_ner
        cmd += ' -loadClassifier %s -outputFormat tabbedEntities' % stanford_classifier
        cmd += ' -textFile %s > %s' % (in_path, out_path)
        # stanford-ner-3.5.1.jar
        # java -mx700m -cp "stanford-ner-3.5.1.jar:" edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ./classifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile %s
        # classifiers/english.all.3class.distsim.crf.ser.gz
        os.system(cmd)


def read_ner_spacy(filenames):
    """read NER output files and store them in a pandas DataFrame"""
    # class Corpus(object):
    #     def __iter__(self):
    #         for parsed in nlp.pipe(
    #             texts=data,
    #             batch_size=5000,
    #             n_threads=cpu_count() - 1 # use as many cores that are available
    #         ):
    #             yield cb(parsed)
    rows = []
    for filename in filenames:
        path = os.path.join(orig_dir, filename)
        if not os.path.exists(path):
            continue

        parsed = nlp(io.open(path, mode='r', encoding='utf-8').read())
        doc_id = filename.split('/')[-1].split('-', 1)[0]

        ner = collections.defaultdict(list)
        for ent in parsed.ents:
            if ent.label_ not in tag_map.keys():
                continue

            ner[ent.sent.start_char].append(dict(
                begin=ent.start_char,
                end=ent.end_char,
                text=ent.text,
                tag=ent.label_
            ))

        for sent_id, sentence in enumerate(parsed.sents):
            # get named entities in this sentence
            ent = ner.get(sentence.start_char, [])

            # if there are more than two, process each combination
            for c in combinations(ent, 2):
                # TODO: should we skip when two NER are immediately next to one another?
                dic = dict(
                    doc_id=doc_id,
                    sent_id=sent_id,
                    sent=sentence.string,
                    subj=c[0]['text'],
                    subj_begin=c[0]['begin'] - sentence.start_char,
                    subj_end=c[0]['end'] - sentence.start_char,
                    subj_tag=c[0]['tag'],
                    obj=c[1]['text'],
                    obj_begin=c[1]['begin'] - sentence.start_char,
                    obj_end=c[1]['end'] - sentence.start_char,
                    obj_tag=c[1]['tag']
                )
                rows.append(dic)
    return pd.DataFrame(rows)


def read_ner_output(filenames):
    """read NER output files and store them in a pandas DataFrame"""
    rows = []
    for filename in filenames:
        path = os.path.join(ner_dir, filename)
        if not os.path.exists(path):
            continue
        with io.open(path, 'r', encoding='utf-8') as f:
            doc_id = filename.split('/')[-1].split('-', 1)[0]
            counter = 0
            tmp = []
            for line in f.readlines():
                # after reading the entire line into 'tmp' process it, the 'line' should be only '\n'
                if len(line.strip()) < 1 and len(tmp) > 2:
                    # get all entries which were tagged with an named entity in our watch list 'tag_map'
                    # this returns the index position of those named entities from 'tmp'
                    ent = [i for i, t in enumerate(tmp) if t[1] in tag_map.keys()]
                    if len(ent) > 2:
                        a = 1
                    # for every combination, of at least 2, add an entry splitting on the named entity boundaries
                    # i.e. if there are only 2, there will be one
                    # 3 = 3, 4 = 6, 5 = 10, 6 = 15, 7 = 21, 8 = 28, etc.
                    for c in combinations(ent, 2):
                        dic = {'sent': u''}
                        dic['doc_id'] = doc_id
                        dic['sent_id'] = counter
                        for j, t in enumerate(tmp):
                            # for the first named entity, treat it as the subject
                            if j == c[0]:
                                if len(dic['sent']) > 0:
                                    dic['subj_begin'] = len(dic['sent']) + 1
                                else:
                                    dic['subj_begin'] = 0

                                if len(dic['sent']) > 0:
                                    dic['subj_end'] = len(dic['sent']) + len(t[0].strip()) + 1
                                else:
                                    dic['subj_end'] = len(t[0].strip())
                                dic['subj'] = t[0].strip()      # named entity string
                                dic['subj_tag'] = t[1].strip()  # tag name of string from NER
                            elif j == c[1]:
                                # for the second named entity, treat it as the object
                                dic['obj_begin'] = len(dic['sent']) + 1
                                dic['obj_end'] = len(dic['sent']) + len(t[0].strip()) + 1
                                dic['obj'] = t[0].strip()       # named entity string
                                dic['obj_tag'] = t[1].strip()   # tag name of string from NER

                            # re-assemble the sentence
                            # 0 = named entity, which is an empty string if nothing was detected
                            if len(dic['sent']) > 0:
                                dic['sent'] += ' ' + t[0].strip()
                            else:
                                dic['sent'] += t[0].strip()

                            # 2 = text following the named entity, which is usually the bulk of the content
                            if len(dic['sent']) > 0:
                                dic['sent'] += ' ' + t[2].strip()
                            else:
                                dic['sent'] += t[2].strip()

                        rows.append(dic)

                    counter += 1
                    tmp = []
                elif len(line.strip()) < 1 and len(tmp) > 0 and len(tmp) <= 2:
                    continue
                elif len(line.strip()) > 0:
                    # extract the contents of the original sentence, as it was before NER
                    # there should be exactly 3 \t for each word from NER, if not pad them
                    # 0 = named entity string
                    # 1 = named entity tag (i.e. ORGANIZATION)
                    # 2 = noun chunk after the named entity
                    e = line.split('\t')
                    if len(e) == 1:
                        e.insert(0, '')
                        e.insert(0, '')
                    if len(e) == 2 and e[1].strip() in tag_map.keys():
                        e.append('')
                    if len(e) != 3:
                        print(e)
                        raise Exception
                    # store running list of the NER output
                    tmp.append(e)
                else:
                    continue

    return pd.DataFrame(rows)


def name2qid(name, tag, alias=False, retry=False):
    """
    find QID (and Freebase ID if given) by name

    >>> name2qid('Barack Obama', 'PERSON')        # perfect match
    ('Q76', '/m/02mjmr')
    >>> name2qid('Obama', 'PERSON', alias=True)   # alias match
    ('Q33687029', '')
    """

    label = 'rdfs:label'
    if alias:
        label = 'skos:altLabel'

    hpCharURL = 'https://query.wikidata.org/sparql?query=\
    SELECT DISTINCT ?item ?fid \
    WHERE {\
    ?item '+label+' "'+name+'"@en.\
    ?item wdt:P31 ?_instanceOf.\
    ?_instanceOf wdt:P279* wd:'+tag_map[tag]+'.\
    OPTIONAL { ?item wdt:P646 ?fid. }\
    }\
    LIMIT 10'
    headers = {"Accept": "application/json"}

    # check response
    r = None
    try:
        r = requests.get(hpCharURL, headers=headers)
    except requests.exceptions.ConnectionError:
        if not retry:
            time.sleep(60)
            name2qid(name, tag, alias, retry=True)
        else:
            return None
    except Exception as e:
        print(e)
        return None

    # check json format
    try:
        response = r.json()
    except ValueError:    # includes JSONDecodeError
        return None

    # parse results
    results = []
    for elm in response['results']['bindings']:
        fid = ''
        if 'fid' in elm:
            fid = elm['fid']['value']
        results.append((elm['item']['value'].split('/')[-1], fid))

    if len(results) < 1:
        return None
    else:
        return results[0]


def search_property(qid1, qid2, retry=False):
    """find property (and schema.org relation if given)

    >>> search_property('Q76', 'Q30') # Q76: Barack Obama, Q30: United States
    [(u'P27', u'country of citizenship', u'nationality')]
    """

    hpCharURL = 'https://query.wikidata.org/sparql?query= \
    SELECT DISTINCT ?p ?l ?s \
    WHERE {\
    wd:'+qid1+' ?p wd:'+qid2+' .\
    ?property ?ref ?p .\
    ?property a wikibase:Property .\
    ?property rdfs:label ?l FILTER (lang(?l) = "en")\
    OPTIONAL { ?property wdt:P1628 ?s FILTER (SUBSTR(str(?s), 1, 18) = "http://schema.org/"). }\
    }\
    LIMIT 10'
    headers = {"Accept": "application/json"}

    # check response
    r = None
    try:
        r = requests.get(hpCharURL, headers=headers)
    except requests.exceptions.ConnectionError:
        if not retry:
            time.sleep(60)
            search_property(qid1, qid2, retry=True)
        else:
            return None
    except Exception as e:
        print(e)
        return None

    # check json format
    try:
        response = r.json()
    except ValueError:
        return None

    # parse results
    results = []
    for elm in response['results']['bindings']:
        schema = ''
        if 's' in elm:
            schema = elm['s']['value'].split('/')[-1]
        results.append((elm['p']['value'].split('/')[-1], elm['l']['value'], schema))

    return results


def slot_filling(qid, pid, tag, retry=False):
    """find slotfiller

    >>> slot_filling('Q76', 'P27', 'LOCATION') # Q76: Barack Obama, P27: country of citizenship
    [('United States', 'Q30', '/m/09c7w0')]
    """

    hpCharURL = 'https://query.wikidata.org/sparql?query=\
    SELECT DISTINCT ?item ?itemLabel ?fid \
    WHERE {\
    wd:'+qid+' wdt:'+pid+' ?item.\
    ?item wdt:P31 ?_instanceOf.\
    ?_instanceOf wdt:P279* wd:'+tag_map[tag]+'.\
    SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }\
    OPTIONAL { ?item wdt:P646 ?fid. }\
    }\
    LIMIT 100'
    headers = {"Accept": "application/json"}

    # check response
    r = None
    try:
        r = requests.get(hpCharURL, headers=headers)
    except requests.exceptions.ConnectionError:
        if not retry:
            time.sleep(60)
            slot_filling(qid, pid, tag, retry=True)
        else:
            return None
    except Exception as e:
        print(e)
        return None

    # check json format
    try:
        response = r.json()
    except ValueError:
        return None

    # parse results
    results = []
    for elm in response['results']['bindings']:
        fid = ''
        if 'fid' in elm:
            fid = elm['fid']['value']
        results.append((elm['itemLabel']['value'], elm['item']['value'].split('/')[-1], fid))

    return results


def loop(step, doc_id, limit, entities, relations, counter):
    """Distant Supervision Loop"""
    # Download wiki articles
    print('[1/4] Downloading wiki articles ...')
    # docs = download_wiki_articles(doc_id, limit)
    docs = os.listdir('./data/orig')
    if docs is None:
        return None

    # Named Entity Recognition
    print('[2/4] Performing named entity recognition ...')
    # exec_ner(docs)
    # wiki_data = read_ner_output(docs)
    path = os.path.join(data_dir, 'candidates%d.tsv' % step)
    if not os.path.isfile(path):
        wiki_data = read_ner_spacy(docs)
        wiki_data.to_csv(path, sep='\t', encoding='utf-8', index=False)
    else:
        wiki_data = pd.read_csv(path, sep='\t', encoding='utf-8')

    doc_id.extend([int(s) for s in wiki_data.doc_id.unique()])

    # Prepare Containers
    unique_entities = set(wiki_data.groupby(['subj', 'subj_tag']).count().index.tolist())
    unique_entities.update(set(wiki_data.groupby(['obj', 'obj_tag']).count().index.tolist()))
    unique_entity_pairs =  set(wiki_data.groupby(['subj', 'obj']).count().index.tolist())
    # for idx, row in wiki_data.iterrows():
    #     unique_entities.add((row['subj'], row['subj_tag']))
    #     unique_entities.add((row['obj'], row['obj_tag']))
    #     unique_entity_pairs.add((row['subj'], row['obj']))

    # Entity Linkage
    print('[3/4] Linking entities ...')
    entities_filename = os.path.join(data_dir, "entities.pickle")
    if os.path.isfile(entities_filename):
        entities = util.load_from_dump(entities_filename)
    else:
        for name, tag in unique_entities:
            if not name in entities and tag in tag_map.keys():
                e = name2qid(name, tag, alias=False)
                if e is None:
                    e = name2qid(name, tag, alias=True)
                entities[name] = e
        util.dump_to_file(entities_filename, entities)

    # Predicate Linkage
    print('[4/4] Linking predicates ...')
    predicates_filename = os.path.join(data_dir, "relations.pickle")
    if os.path.isfile(predicates_filename):
        relations = util.load_from_dump(predicates_filename)
    else:
        for subj, obj in unique_entity_pairs:
            if not (subj, obj) in relations:
                if entities.get(subj) is not None and entities.get(obj) is not None:
                    if (entities[subj][0] != entities[obj][0]) or (subj != obj):
                        arg1 = entities[subj][0]
                        arg2 = entities[obj][0]
                        relations[(subj, obj)] = search_property(arg1, arg2)
                        #TODO: alternative name relation
                        #elif (entities[subj][0] == entities[obj][0]) and (subj != obj):
                        #    relations[(subj, obj)] = 'P'
        util.dump_to_file(predicates_filename, relations)

    # Assign relation
    # i.e. extract the 'class' name for this relationship
    wiki_data['rel'] = pd.Series(index=wiki_data.index, dtype=str)
    rel = list(map(lambda x: ', '.join(set([s[0] for s in x])), relations.values()))
    for i, r in enumerate(relations):
        if len(rel[i]) > 0:
            # counter += 1
            idx = (wiki_data['subj'] == r[0]) & (wiki_data['obj'] == r[1])
            wiki_data.loc[idx, 'rel'] = rel[i]


    # Save
    path = os.path.join(data_dir, 'candidates%d.tsv' % step)
    wiki_data.to_csv(path, sep='\t', encoding='utf-8', index=False)

    # Cleanup
    # for f in glob.glob(os.path.join(orig_dir, '*')):
    #     os.remove(f)
    #
    # for f in glob.glob(os.path.join(ner_dir, '*')):
    #     os.remove(f)

    return doc_id, entities, relations, counter


def extract_relations(entities, relations):
    """extract relations"""
    rows = []
    for k, v in relations.items():
        if v is not None and len(v) > 0:
            for r in v:
                dic = {}
                dic['subj_qid'] = entities[k[0]][0]
                dic['subj_fid'] = entities[k[0]][1]
                dic['subj'] = k[0]
                dic['obj_qid'] = entities[k[1]][0]
                dic['obj_fid'] = entities[k[1]][1]
                dic['obj'] = k[1]
                dic['rel_id'] = r[0]
                dic['rel'] = r[1]
                dic['rel_schema'] = r[2]
                #TODO: add number of mentions
                #dic['wikidata_idx'] = entity_pairs[k]
                rows.append(dic)
    return pd.DataFrame(rows)


def positive_examples():
    entities = {}
    relations = {}
    counter = 0
    limit = 1000
    doc_id = []
    step = 1

    if not os.path.exists(orig_dir):
        os.makedirs(orig_dir)
    if not os.path.exists(ner_dir):
        os.makedirs(ner_dir)

    #for j in range(1, step):
    #    wiki_data = pd.read_csv(os.path.join(data_dir, "candidates%d.tsv" % j), sep='\t', index_col=0)
    #    doc_id.extend([int(s) for s in wiki_data.doc_id.unique()])
    #    counter += int(wiki_data.rel.count())

    while counter < 10000 and step < 100:
        print('===== step %d =====' % step)
        ret = loop(step, doc_id, limit, entities, relations, counter)
        if ret is not None:
            doc_id, entities, relations, counter = ret

        step += 1

    # positive candidates
    # extract all observations from the wiki crawl which have at least one relationship (i.e. 'class')
    positive_data = []
    for f in glob.glob(os.path.join(data_dir, 'candidates*.tsv')):
        pos = pd.read_csv(f, sep='\t', encoding='utf-8')
        positive_data.append(pos[pd.notnull(pos.rel)])
    positive_df = pd.concat(positive_data, axis=0, ignore_index=True)
    positive_df[col].to_csv(os.path.join(data_dir, 'positive_candidates.tsv'), sep='\t', encoding='utf-8', index=False)

    # save relations
    pos_rel = extract_relations(entities, relations)
    pos_rel.to_csv(os.path.join(data_dir, 'positive_relations.tsv'), sep='\t', encoding='utf-8', index=False)


def negative_examples():
    negative = {}

    unique_pair = set([])
    neg_candidates = []

    #TODO: replace with positive_relations.tsv
    entities = util.load_from_dump(os.path.join(data_dir, "entities.pickle"))
    relations = util.load_from_dump(os.path.join(data_dir, "relations.pickle"))

    rel_counter = Counter([u[0] for r in relations.values() if r is not None and len(r) > 0 for u in r])
    most_common_rel = [r[0] for r in rel_counter.most_common(10)]


    for data_path in glob.glob(os.path.join(data_dir, 'candidates*.tsv')):
        neg = pd.read_csv(data_path, sep='\t', encoding='utf-8')
        negative_df = neg[pd.isnull(neg.rel)]

        # Assign relation
        for idx, row in negative_df.iterrows():
            if (
                row['subj'] in entities and entities[row['subj']] is not None
                and row['obj'] in entities and entities[row['obj']] is not None
            ):
                qid = entities[row['subj']][0]
                target = entities[row['obj']][0]
                candidates = []
                for pid in most_common_rel:
                    if (qid, pid) not in unique_pair:
                        unique_pair.add((qid, pid))
                        items = slot_filling(qid, pid, row['obj_tag'])
                        if items is not None and len(items) > 1:
                            qids = [q[1] for q in items]
                            if target not in qids:
                                candidates.append(pid)

                if len(candidates) > 0:
                    row['rel'] = ', '.join(candidates)
                    neg_candidates.append(row)


    # extract all observations from the wiki crawl which have at least one relationship (i.e. 'class')
    neg_examples = pd.DataFrame(neg_candidates)
    neg_examples[col].to_csv(os.path.join(data_dir, 'negative_candidates.tsv'), sep='\t', encoding='utf-8', index=False)


    # save relations
    #pos_rel = extract_relations(entities, negative)
    #pos_rel.to_csv(os.path.join(data_dir, 'negative_relations.tsv'), sep='\t', encoding='utf-8')


def load_gold_patterns():
    def clean_str(string):
        string = re.sub(r", ", " , ", string)
        string = re.sub(r"' ", " ' ", string)
        string = re.sub(r" \* ", " .* ", string)
        string = re.sub(r"\(", "-LRB-", string)
        string = re.sub(r"\)", "-RRB-", string)
        string = re.sub(r" {2,}", " ", string)
        return string.strip()

    g_patterns = []
    g_labels = []
    with io.open(os.path.join(data_dir, 'gold_patterns.tsv'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 0 and not line.startswith('#'):
                e = line.split('\t', 1)
                if len(e) > 1:
                    g_patterns.append(clean_str(e[1]))
                    g_labels.append(e[0])
                else:
                    print(e)
                    raise Exception('Process Error: %s' % os.path.join(data_dir, 'gold_patterns.tsv'))

    return pd.DataFrame({'pattern': g_patterns, 'label': g_labels})


def score_reliability(gold_patterns, sent, rel, subj, obj):
    for name, group in gold_patterns.groupby('label'):
        if name in [r.strip() for r in rel.split(',')]:
            for i, g in group.iterrows():
                pattern = g['pattern']
                pattern = re.sub(r'\$ARG(0|1)', subj, pattern, count=1)
                pattern = re.sub(r'\$ARG(0|2)', obj, pattern, count=1)
                match = re.search(pattern, sent)
                if match:
                    return 1.0
    return 0.0


def extract_positive():
    if not os.path.exists(os.path.join(data_dir, 'mlmi')):
        os.makedirs(os.path.join(data_dir, 'mlmi'))
    if not os.path.exists(os.path.join(data_dir, 'er')):
        os.makedirs(os.path.join(data_dir, 'er'))

    # read gold patterns to extract attention
    gold_patterns = load_gold_patterns()


    #TODO: replace with negative_relations.tsv
    entities = util.load_from_dump(os.path.join(data_dir, "entities.pickle"))
    relations = util.load_from_dump(os.path.join(data_dir, "relations.pickle"))

    # count the number of times each 'class' appeared
    # take the 50 most common class combinations, and filter out the relations which occur less than n times
    top_n = 0
    rel_c = Counter([u[0] for r in relations.values() if r is not None and len(r) > 0 for u in r])
    rel_c_top = [k for k, v in rel_c.most_common(50) if v >= top_n]

    # positive examples
    positive_df = pd.read_csv(os.path.join(data_dir, 'positive_candidates.tsv'),
                              sep='\t', encoding='utf-8', index_col=0)

    positive_df['right'] = pd.Series(index=positive_df.index, dtype=str)
    positive_df['middle'] = pd.Series(index=positive_df.index, dtype=str)
    positive_df['left'] = pd.Series(index=positive_df.index, dtype=str)
    positive_df['clean'] = pd.Series(index=positive_df.index, dtype=str)
    positive_df['label'] = pd.Series(index=positive_df.index, dtype=str)
    positive_df['attention'] = pd.Series([0.0]*len(positive_df), index=positive_df.index, dtype=np.float32)

    num_er = 0
    with io.open(os.path.join(data_dir, 'er', 'source.txt'), 'w', encoding='utf-8') as f:
        for idx, row in positive_df.iterrows():

            # restore relation
            rel = ['<' + l.strip() + '>' for l in row['rel'].split(',') if l.strip() in rel_c_top]
            if len(rel) > 0:

                s = row['sent']
                subj = '<' + entities.get(row['subj'], (row['subj'], ))[0] + '>'
                obj = '<' + entities.get(row['obj'], (row['obj'], ))[0] + '>'
                left = s[:row['subj_begin']] + subj
                middle = s[row['subj_end']:row['obj_begin']]
                right = obj + s[row['obj_end']:]
                text = left.strip() + ' ' + middle.strip() + ' ' + right.strip()

                # check if begin-end position is correct
                assert s[row['subj_begin']:row['subj_end']] == row['subj']
                assert s[row['obj_begin']:row['obj_end']] == row['obj']

                # MLMI dataset
                # filter out too long sentences
                if len(left.split()) < 100 and len(middle.split()) < 100 and len(right.split()) < 100:

                    positive_df.set_value(idx, 'right', right.strip())
                    positive_df.set_value(idx, 'middle', middle.strip())
                    positive_df.set_value(idx, 'left', left.strip())
                    positive_df.set_value(idx, 'clean', text.strip())

                    # binarize label
                    label = ['0'] * len(rel_c_top)
                    for u in row['rel'].split(','):
                        if u.strip() in rel_c_top:
                            label[rel_c_top.index(u.strip())] = '1'
                    positive_df.set_value(idx, 'label', ' '.join(label))

                    # score reliability if positive match on any known token patterns
                    # TODO: is this mainly used for the distant supervision training set build?
                    # https://github.com/beroth/relationfactory/blob/master/resources/manual_annotation/context_patterns2012.txt
                    reliability = score_reliability(gold_patterns, s, row['rel'], row['subj'], row['obj'])
                    positive_df.set_value(idx, 'attention', reliability)

                # ER dataset
                for r in rel:
                    num_er += 1
                    f.write(subj + ' ' + r + ' ' + obj + '\n')

    with io.open(os.path.join(data_dir, 'er', 'target.txt'), mode='w', encoding='utf-8') as f:
        for _ in range(num_er):
            f.write('1 0\n')

    positive_df_valid = positive_df[pd.notnull(positive_df.clean)]
    assert len(positive_df_valid['clean']) == len(positive_df_valid['label'])

    positive_df_valid['right'].to_csv(os.path.join(data_dir, 'mlmi', 'source.right'),
                                      sep='\t', index=False, header=False, encoding='utf-8')
    positive_df_valid['middle'].to_csv(os.path.join(data_dir, 'mlmi', 'source.middle'),
                                       sep='\t', index=False, header=False, encoding='utf-8')
    positive_df_valid['left'].to_csv(os.path.join(data_dir, 'mlmi', 'source.left'),
                                     sep='\t', index=False, header=False, encoding='utf-8')
    positive_df_valid['clean'].to_csv(os.path.join(data_dir, 'mlmi', 'source.txt'),
                                      sep='\t', index=False, header=False, encoding='utf-8')
    positive_df_valid['label'].to_csv(os.path.join(data_dir, 'mlmi', 'target.txt'),
                                      sep='\t', index=False, header=False, encoding='utf-8')
    positive_df_valid['attention'].to_csv(os.path.join(data_dir, 'mlmi', 'source.att'),
                                          sep='\t', index=False, header=False, encoding='utf-8')


def extract_negative():
    entities = util.load_from_dump(os.path.join(data_dir, "entities.pickle"))

    # negative examples
    negative_df = pd.read_csv(os.path.join(data_dir, 'negative_candidates.tsv'),
                              sep='\t', encoding='utf-8', index_col=0)

    with io.open(os.path.join(data_dir, 'er', 'source.txt'), 'a', encoding='utf-8') as source_file:
        with io.open(os.path.join(data_dir, 'er', 'target.txt'), 'a', encoding='utf-8') as target_file:
            for idx, row in negative_df.iterrows():
                s = row['sent']

                subj = '<' + entities[row['subj'].encode('utf-8')][0] + '>'
                obj = '<' + entities[row['obj'].encode('utf-8')][0] + '>'
                rel = ['<' + l.strip() + '>' for l in row['rel'].split(',')]

                assert s[row['subj_begin']:row['subj_end']] == row['subj']
                assert s[row['obj_begin']:row['obj_end']] == row['obj']

                if len(rel) > 0:
                    for r in rel:
                        source_file.write(subj + ' ' + r + ' ' + obj + '\n')
                        target_file.write('0 1\n')


def main():
    # gather positive examples
    if not os.path.exists(os.path.join(data_dir, 'positive_candidates.tsv')):
        positive_examples()
    extract_positive()

    # gather negative examples
    if not os.path.exists(os.path.join(data_dir, 'negative_candidates.tsv')):
        negative_examples()
    extract_negative()


if __name__ == '__main__':
    main()
