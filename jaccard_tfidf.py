from math import *
import re
import string
document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin was found to be riding a horse, again, without a shirt on while hunting deer. Vladimir Putin always seems so serious about things - even riding horses."

all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]
tokenize = lambda doc: doc.lower().split(" ")
tokenized_documents = [tokenize(d) for d in all_documents] # tokenized docs
all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
#for idx,document in enumerate(document_list):
#	document_list[idx] = re.sub(r'[^\w\s]','',document_list[idx])

#tokenized_doc = [tokenize(d) for d in document_list]
#all_token_set = set([item for doc in tokenized_doc for item in doc])

def jaccard_similarity(doc1,doc2):
	intersection = set(doc1).intersection(set(doc2))
	union = set(doc1).union(set(doc2))
	return len(intersection) / float(len(union))

#### TF-IDF (term frequency - inverse document frequency)

def term_frequency(token,document):
	return document.count(token)

def sub_linear_frequency(term,document):
	count = document.count(term)
	if count ==0 :
		return 0
	return 1+log(count)

### word weights
def inverse_document_frequency(document):
	idf_values = {}
	all_token_set = set([item for doc in document for item in doc])
	for token in all_token_set:
		contains_token = map(lambda doc: token in doc , document)
		idf_values[token] = 1 + log(len(document)/(sum(contains_token)))
	return idf_values

def tfidf(document):
	tokenized = [tokenize(doc) for doc in document]
	idf = inverse_document_frequency(tokenized)
	tfidf_document = []
	for doc in tokenized:
		tfidf_tmp = []
		for term in idf.keys():
			tf = sub_linear_frequency(term,doc) 
			tfidf_tmp.append(tf * idf[term])
		tfidf_document.append(tfidf_tmp)
	return tfidf_document

if __name__ == "__main__":
	print('Jaccard Similarity : {}'.format(jaccard_similarity(tokenized_documents[2],tokenized_documents[4])))
	idf_values = inverse_document_frequency(tokenized_documents)
	print(idf_values['the'])
	#tfidf_representation = tfidf(document_list)
	#print tfidf_representation[0], document_0
	#tfidf_rep = tfidf(document_list)
	#print(tfidf_rep[0],document_list[0])






