import spacy
import timeit
import math
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from collections import Counter
from lxml import etree
from glob import glob
from unicodedata import normalize

def lemmatize(path):
    list_lemma = []
    with open(path, encoding="utf8") as file:
        tree = etree.parse(file)
        tag = est_canon(tree)
        if tag == True:
            print("canon")
        else:
            print("non_canon")
        if tree.findall(".//p"):
            for paragraphe in tree.findall(".//p"):
                if paragraphe.text:
                    clean_text = normalize("NFKD", paragraphe.text)
                    docs = nlp(clean_text)
                    for token in docs:
                        if token.pos_ != "PUNCT" and "SPACE" and "X" and "SYM":
                            list_lemma.append(token.lemma_)
    return list_lemma, tag


def est_canon(tree):
    if tree.find(".//profileDesc") is not None:
        profil = tree.find(".//profileDesc")
        if profil.get("tag") == "canon":
            return True
        else:
            return False

def bigrammize(list_lemma):
    list_bigram = []
    for indice_lemma in range(len(list_lemma)-1):
        bigram = list_lemma[indice_lemma]+'_'+list_lemma[indice_lemma+1]
        list_bigram.append(bigram)
    return list_bigram

def rollingnwords(list_lemma, n):
    i = 0 # i stocke l'indice auquel on est dans le rolling
    list_rolling = []
    while i-n < len(list_lemma):
        list_rolling.append(list_lemma[i:i+n])
        i+=n
    return list_rolling

def rolling_type_token(rolling_list_lemma, window):
    list_rolling_type_token = []
    for list_lemma in rolling_list_lemma:
        lemmes_freq = Counter()
        for lemma in list_lemma:
            lemmes_freq[lemma] += 1
        if sum(lemmes_freq.values()) == window:
            list_rolling_type_token.append(round(len(lemmes_freq)/sum(lemmes_freq.values()),2))
    return list_rolling_type_token

def rolling_shannon(rolling_list_bigram):
    shannon_measures = []
    for list_bigram in rolling_list_bigram:
        shannon_sum = 0 # initialisation de l'indice de shannon
        dict_conteur = Counter(list_bigram)
        for bigram in list_bigram:
            # on recupere la proportion pi de chaque bigram par rapport à tous les autres bigrams
            prop = dict_conteur[bigram]/len(list_bigram)
            shannon_courant = prop * (math.log(prop, 2))
            # on met à jour l'indice de shannon
            shannon_sum += shannon_courant
        shannon_measures.append(round(shannon_sum * -1,2))
    return shannon_measures


def moulinette(path_name, window):

    canon = False

    annee_canon = []
    annee_archive = []

    type_token_canon_df = pd.DataFrame()
    type_token_archive_df = pd.DataFrame()

    shannon_canon_df = pd.DataFrame()
    shannon_archive_df = pd.DataFrame()

    for doc in glob(path_name):
        doc_name = path.splitext(path.basename(doc))[0]
        date = doc_name.split("_")[0]
        print(doc_name)

        list_lemma, canon = lemmatize(doc)

        rolling_list_lemma = rollingnwords(list_lemma, window)
        type_token = rolling_type_token(rolling_list_lemma, window)

        list_bigram = bigrammize(list_lemma)
        rolling_list_bigram = rollingnwords(list_bigram, window)

        indice_shannon = rolling_shannon(rolling_list_bigram)

        if canon:
            t_canon = pd.Series(type_token, name=doc_name)
            type_token_canon_df = pd.concat([type_token_canon_df, t_canon], axis=1)
            s_canon = pd.Series(indice_shannon, name=doc_name)
            shannon_canon_df = pd.concat([shannon_canon_df, s_canon], axis=1)
            annee_canon.append(date)
        else:
            t_archive = pd.Series(type_token, name=doc_name)
            type_token_archive_df = pd.concat([type_token_archive_df, t_archive], axis=1)
            s_archive = pd.Series(indice_shannon, name=doc_name)
            shannon_archive_df = pd.concat([shannon_archive_df, s_archive], axis=1)
            annee_archive.append(date)

    return annee_canon, annee_archive, type_token_canon_df, type_token_archive_df, shannon_canon_df, shannon_archive_df
