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
import time
import requests
import urllib
import glob
from codecs import open
from itertools import combinations
from collections import Counter

import util

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


# global variables
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
orig_dir = os.path.join(data_dir, 'orig')
ner_dir = os.path.join(data_dir, 'ner')


ner_path = "/usr/local/Cellar/stanford-ner/3.5.2/libexec/"
stanford_classifier = os.path.join(ner_path, 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
stanford_ner = os.path.join(ner_path, 'stanford-ner.jar')

tag_map = {
    'ORGANIZATION': 'Q43229',  # https://www.wikidata.org/wiki/Q43229
    'LOCATION': 'Q17334923',   # https://www.wikidata.org/wiki/Q17334923
    'PERSON': 'Q5'             # https://www.wikidata.org/wiki/Q5
}

# column names in DataFrame
col = ['doc_id', 'sent_id', 'sent', 'subj', 'subj_begin', 'subj_end', 'subj_tag',
       'rel', 'obj', 'obj_begin', 'obj_end', 'obj_tag']


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
        r = urllib.urlopen(query).read()
    except Exception as e:
        if not retry:
            download_wiki_articles(doc_id, limit, retry=True)
        else:
            print(e.message)
            return None
    pages = bs(r, "html.parser").findAll('page')
    if len(pages) < 1:
        return None
    docs = []
    for page in pages:
        if int(page['id']) in doc_id:
            continue

        link = base_path + "&prop=revisions&pageids=%s&rvprop=content&rvparse" % page['id']
        content = urllib.urlopen(link).read()
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
            with open(os.path.join(orig_dir, filename), 'w', encoding='utf-8') as f:
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
        os.system(cmd)


def read_ner_output(filenames):
    """read NER output files and store them in a pandas DataFrame"""
    rows = []
    for filename in filenames:
        path = os.path.join(ner_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            doc_id = filename.split('/')[-1].split('-', 1)[0]
            counter = 0
            tmp = []
            for line in f.readlines():
                if len(line.strip()) < 1 and len(tmp) > 2:
                    ent = [i for i, t in enumerate(tmp) if t[1] in tag_map.keys()]
                    for c in combinations(ent, 2):
                        dic = {'sent': u''}
                        dic['doc_id'] = doc_id
                        dic['sent_id'] = counter
                        for j, t in enumerate(tmp):
                            if j == c[0]:
                                if len(dic['sent']) > 0:
                                    dic['subj_begin'] = len(dic['sent']) + 1
                                else:
                                    dic['subj_begin'] = 0
                                if len(dic['sent']) > 0:
                                    dic['subj_end'] = len(dic['sent']) + len(t[0].strip()) + 1
                                else:
                                    dic['subj_end'] = len(t[0].strip())
                                dic['subj'] = t[0].strip()
                                dic['subj_tag'] = t[1].strip()
                            elif j == c[1]:
                                dic['obj_begin'] = len(dic['sent']) + 1
                                dic['obj_end'] = len(dic['sent']) + len(t[0].strip()) + 1
                                dic['obj'] = t[0].strip()
                                dic['obj_tag'] = t[1].strip()

                            if len(dic['sent']) > 0:
                                dic['sent'] += ' ' + t[0].strip()
                            else:
                                dic['sent'] += t[0].strip()
                            if len(dic['sent']) > 0:
                                dic['sent'] += ' ' + t[2].strip()
                            else:
                                dic['sent'] += t[2].strip()
                                #print('"'+dic['sent']+'"', len(dic['sent']))
                        rows.append(dic)
                        #print(dic)
                    counter += 1
                    tmp = []
                elif len(line.strip()) < 1 and len(tmp) > 0 and len(tmp) <= 2:
                    continue
                elif len(line.strip()) > 0:
                    e = line.split('\t')
                    if len(e) == 1:
                        e.insert(0, '')
                        e.insert(0, '')
                    if len(e) == 2 and e[1].strip() in tag_map.keys():
                        e.append('')
                    if len(e) != 3:
                        print(e)
                        raise Exception
                    tmp.append(e)
                else:
                    continue

    return pd.DataFrame(rows)


def name2qid(name, tag, alias=False, retry=False):
    """find QID (and Freebase ID if given) by name

    >>> name2qid('Barack Obama', 'PERSON')        # perfect match
    (u'Q76', u'/m/02mjmr')
    >>> name2qid('Obama', 'PERSON', alias=True)   # alias match
    (u'Q76', u'/m/02mjmr')
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
        print(e.message)
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
        if elm.has_key('fid'):
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
        print(e.message)
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
        if elm.has_key('s'):
            schema = elm['s']['value'].split('/')[-1]
        results.append((elm['p']['value'].split('/')[-1], elm['l']['value'], schema))

    return results


def slot_filling(qid, pid, tag, retry=False):
    """find slotfiller

    >>> slot_filling('Q76', 'P27', 'LOCATION') # Q76: Barack Obama, P27: country of citizenship
    [(u'United States', u'Q30', u'/m/09c7w0')]
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
        print(e.message)
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
        if elm.has_key('fid'):
            fid = elm['fid']['value']
        results.append((elm['itemLabel']['value'], elm['item']['value'].split('/')[-1], fid))

    return results


def loop(step, doc_id, limit, entities, relations, counter):
    """Distant Supervision Loop"""
    # Download wiki articles
    print('[1/4] Downloading wiki articles ...')
    docs = download_wiki_articles(doc_id, limit)
    if docs is None:
        return None

    # Named Entity Recognition
    print('[2/4] Performing named entity recognition ...')
    exec_ner(docs)
    wiki_data = read_ner_output(docs)
    path = os.path.join(data_dir, 'candidates%d.tsv' % step)
    wiki_data.to_csv(path, sep='\t', encoding='utf-8')
    doc_id.extend([int(s) for s in wiki_data.doc_id.unique()])

    # Prepare Containers
    unique_entities = set([])
    unique_entity_pairs = set([])
    for idx, row in wiki_data.iterrows():
        unique_entities.add((row['subj'], row['subj_tag']))
        unique_entities.add((row['obj'], row['obj_tag']))
        unique_entity_pairs.add((row['subj'], row['obj']))

    # Entity Linkage
    print('[3/4] Linking entities ...')
    for name, tag in unique_entities:
        if not entities.has_key(name) and tag in tag_map.keys():
            e = name2qid(name, tag, alias=False)
            if e is None:
                e = name2qid(name, tag, alias=True)
            entities[name] = e
    util.dump_to_file(os.path.join(data_dir, "entities.cPickle"), entities)

    # Predicate Linkage
    print('[4/4] Linking predicates ...')
    for subj, obj in unique_entity_pairs:
        if not relations.has_key((subj, obj)):
            if entities[subj] is not None and entities[obj] is not None:
                if (entities[subj][0] != entities[obj][0]) or (subj != obj):
                    arg1 = entities[subj][0]
                    arg2 = entities[obj][0]
                    relations[(subj, obj)] = search_property(arg1, arg2)
                    #TODO: alternative name relation
                    #elif (entities[subj][0] == entities[obj][0]) and (subj != obj):
                    #    relations[(subj, obj)] = 'P'
    util.dump_to_file(os.path.join(data_dir, "relations.cPickle"), relations)

    # Assign relation
    wiki_data['rel'] = pd.Series(index=wiki_data.index, dtype=str)
    for idx, row in wiki_data.iterrows():
        entity_pair = (row['subj'], row['obj'])

        if relations.has_key(entity_pair):
            rel = relations[entity_pair]
            if rel is not None and len(rel) > 0:
                counter += 1
                wiki_data.set_value(idx, 'rel', ', '.join(set([s[0] for s in rel])))
    # Save
    path = os.path.join(data_dir, 'candidates%d.tsv' % step)
    wiki_data.to_csv(path, sep='\t', encoding='utf-8')

    # Cleanup
    for f in glob.glob(os.path.join(orig_dir, '*')):
        os.remove(f)
    for f in glob.glob(os.path.join(ner_dir, '*')):
        os.remove(f)

    return doc_id, entities, relations, counter


def extract_relations(entities, relations):
    """extract relations"""
    rows = []
    for k, v in relations.iteritems():
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
        os.mkdir(orig_dir)
    if not os.path.exists(ner_dir):
        os.mkdir(ner_dir)

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
    positive_data = []
    for f in glob.glob(os.path.join(data_dir, 'candidates*.tsv')):
        pos = pd.read_csv(f, sep='\t', encoding='utf-8', index_col=0)
        positive_data.append(pos[pd.notnull(pos.rel)])
    positive_df = pd.concat(positive_data, axis=0, ignore_index=True)
    positive_df[col].to_csv(os.path.join(data_dir, 'positive_candidates.tsv'), sep='\t', encoding='utf-8')

    # save relations
    pos_rel = extract_relations(entities, relations)
    pos_rel.to_csv(os.path.join(data_dir, 'positive_relations.tsv'), sep='\t', encoding='utf-8')


def negative_examples():
    negative = {}

    unique_pair = set([])
    neg_candidates = []

    #TODO: replace with positive_relations.tsv
    entities = util.load_from_dump(os.path.join(data_dir, "entities.cPickle"))
    relations = util.load_from_dump(os.path.join(data_dir, "relations.cPickle"))

    rel_counter = Counter([u[0] for r in relations.values() if r is not None and len(r) > 0 for u in r])
    most_common_rel = [r[0] for r in rel_counter.most_common(10)]


    for data_path in glob.glob(os.path.join(data_dir, 'candidates*.tsv')):
        neg = pd.read_csv(data_path, sep='\t', encoding='utf-8', index_col=0)
        negative_df = neg[pd.isnull(neg.rel)]

        # Assign relation
        for idx, row in negative_df.iterrows():
            if (entities.has_key(row['subj']) and entities[row['subj']] is not None \
                        and entities.has_key(row['obj']) and entities[row['obj']] is not None):
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


    neg_examples = pd.DataFrame(neg_candidates)
    neg_examples[col].to_csv(os.path.join(data_dir, 'negative_candidates.tsv'), sep='\t', encoding='utf-8')


    # save relations
    #pos_rel = extract_relations(entities, negative)
    #pos_rel.to_csv(os.path.join(data_dir, 'positive_relations.tsv'), sep='\t', encoding='utf-8')


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
    with open(os.path.join(data_dir, 'gold_patterns.tsv'), 'r') as f:
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
        os.mkdir(os.path.join(data_dir, 'mlmi'))
    if not os.path.exists(os.path.join(data_dir, 'er')):
        os.mkdir(os.path.join(data_dir, 'er'))

    # read gold patterns to extract attention
    gold_patterns = load_gold_patterns()


    #TODO: replace with negative_relations.tsv
    entities = util.load_from_dump(os.path.join(data_dir, "entities.cPickle"))
    relations = util.load_from_dump(os.path.join(data_dir, "relations.cPickle"))

    # filter out the relations which occur less than 50 times
    rel_c = Counter([u[0] for r in relations.values() if r is not None and len(r) > 0 for u in r])
    rel_c_top = [k for k, v in rel_c.most_common(50) if v >= 50]

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
    with open(os.path.join(data_dir, 'er', 'source.txt'), 'w', encoding='utf-8') as f:
        for idx, row in positive_df.iterrows():

            # restore relation
            rel = ['<' + l.strip() + '>' for l in row['rel'].split(',') if l.strip() in rel_c_top]
            if len(rel) > 0:

                s = row['sent']
                subj = '<' + entities[row['subj'].encode('utf-8')][0] + '>'
                obj = '<' + entities[row['obj'].encode('utf-8')][0] + '>'
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

                    # score reliability if positive
                    reliability = score_reliability(gold_patterns, s, row['rel'], row['subj'], row['obj'])
                    positive_df.set_value(idx, 'attention', reliability)

                # ER dataset
                for r in rel:
                    num_er += 1
                    f.write(subj + ' ' + r + ' ' + obj + '\n')

    with open(os.path.join(data_dir, 'er', 'target.txt'), 'w', encoding='utf-8') as f:
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
    entities = util.load_from_dump(os.path.join(data_dir, "entities.cPickle"))

    # negative examples
    negative_df = pd.read_csv(os.path.join(data_dir, 'negative_candidates.tsv'),
                              sep='\t', encoding='utf-8', index_col=0)

    with open(os.path.join(data_dir, 'er', 'source.txt'), 'a', encoding='utf-8') as source_file:
        with open(os.path.join(data_dir, 'er', 'target.txt'), 'a', encoding='utf-8') as target_file:
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
