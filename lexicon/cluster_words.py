from nltk.corpus import wordnet as wn
print("loaded")

def get_synonyms(word):
    synonyms = set()
    for ss in wn.synsets(word):
        synonyms |= set(ss.lemma_names())
    return synonyms

def get_leader(word, word_clusters):
    if "pointer" not in word_clusters[word]:
        return word
    return get_leader(word_clusters[word]["pointer"], word_clusters)

def cluster_words_by_synonym(word_list):
    word_clusters = {}
    for i in range(len(word_list)):
        word = word_list[i]
        word_clusters[word] = {"cluster":{word}, "synonyms":get_synonyms(word)}
    for i in range(len(word_list)-1):
        word = word_list[i]
        word_source = get_leader(word, word_clusters)
        for j in range(i+1, len(word_list)):
            word_next = word_list[j]
            word_next_source = get_leader(word_next, word_clusters)
            if word_next_source in word_clusters[word_source]["synonyms"] and word_next_source not in word_clusters[word_source]["cluster"]:
                word_clusters[word_source]["cluster"] |= word_clusters[word_next_source]["cluster"]
                word_clusters[word_source]["synonyms"] |= word_clusters[word_next_source]["synonyms"]
                word_clusters[word_next_source]["pointer"] = word_source
    res = []
    for word in word_clusters:
        if "pointer" not in word_clusters[word]:
            res.append(word_clusters[word]["cluster"])
    return res

if __name__ == "__main__":
    #print(get_synonyms("sleep"))
    #print(cluster_words_by_synonym(["dog", "chase", "sleep"]))
    #print(get_synonyms("tree"))
    #print(get_synonyms("corner"))
    #print(get_synonyms("niche"))
    #print(get_synonyms("tune"))
    #print(get_synonyms("melody"))
    #print(cluster_words_by_synonym(["tree", "tune", "corner", "niche", "apple", "melody"]))
    print(cluster_words_by_synonym(["tree", "corner", "niche", "apple", "tune", "melody"]))
