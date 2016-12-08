# http://dataconomy.com/implementing-the-five-most-popular-similarity-measures-in-python/
from math import *
## Euclidean Distance
def euclidean_distance(x,y):
	print('Computing Euclidean Distance.....')
	return sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))

## Manhattan distance
def manhattan_distance(x,y):
	print('Computing Manhattan Distance .....')
	return sum(abs(a-b) for a,b in zip(x,y))


def square_root(x):

	return round(sqrt(sum([a*a for a in x])),3)

## Cosine similarity
def consine_similarity(x,y):


	print('Computing Consine Similarity ....')
	## sim(A,b) = cos(theta) = (A dot B) / (norm A * norm B)
	numerator = sum(a*b for a,b in zip(x,y))
	denominator = square_root(x) * square_root(y)
	return round(numerator / float(denominator),3)

## Jaccard similarity
def jaccard_similarity(x,y):
	## the number elements of intersection(x,y) / the number elements of union(x,y)
	print('Computing Jaccard Similarity ....')
	intersection = len(set.intersection(set(x),set(y)))
	union = len(set.union(set(x),set(y)))
	return intersection / float(union)


if __name__ == '__main__':
	print('Euclidean Distance : {}'.format(euclidean_distance([0,3,4,5],[7,6,3,-1])))
	print('Manhattan Distance : {}'.format(manhattan_distance([10,20,10],[10,20,20])))
	print('Consine Similarity : {}'.format(consine_similarity([3, 45, 7, 2], [2, 54, 13, 15])))
	print('Jaccard Similarity : {}'.format(jaccard_similarity([0,1,2,5,6],[0,2,3,5,7,9])))


