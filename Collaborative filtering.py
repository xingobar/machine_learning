from math import *
dataset={
 'Lisa Rose': {
 'Lady in the Water': 2.5,
 'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0,
 'Superman Returns': 3.5,
 'You, Me and Dupree': 2.5,
 'The Night Listener': 3.0},
 'Gene Seymour': {'Lady in the Water': 3.0,
 'Snakes on a Plane': 3.5,
 'Just My Luck': 1.5,
 'Superman Returns': 5.0,
 'The Night Listener': 3.0,
 'You, Me and Dupree': 3.5},
 
 'Michael Phillips': {'Lady in the Water': 2.5,
 'Snakes on a Plane': 3.0,
 'Superman Returns': 3.5,
 'The Night Listener': 4.0},
 'Claudia Puig': {'Snakes on a Plane': 3.5,
 'Just My Luck': 3.0,
 'The Night Listener': 4.5,
 'Superman Returns': 4.0,
 'You, Me and Dupree': 2.5},
 
 'Mick LaSalle': {'Lady in the Water': 3.0,
 'Snakes on a Plane': 4.0,
 'Just My Luck': 2.0,
 'Superman Returns': 3.0,
 'The Night Listener': 3.0,
 'You, Me and Dupree': 2.0},
 
 'Jack Matthews': {'Lady in the Water': 3.0,
 'Snakes on a Plane': 4.0,
 'The Night Listener': 3.0,
 'Superman Returns': 5.0,
 'You, Me and Dupree': 3.5},
 
 'Toby': {'Snakes on a Plane':4.5,
 'You, Me and Dupree':1.0,
 'Superman Returns':4.0}}

def perason_correlation(person1,person2):

	###### Formula ############
	## Sxx = sigma(x^2) - (sigma(x))^2  /n
	## Syy = sigma(y^2) - (sigma(y))^2 /n
	## Sxy = sigma(xy) - (sigma(x) * sigma(y)) /n
	## r = Sxy / sqrt(Sxx * Syy)
	###########################

	both_rated = {}
	for idx,(name,rank) in enumerate(dataset[person1].items()):
		if name in dataset[person2]:
			both_rated[name] = 1
	number_of_rated  = len(both_rated)
	if number_of_rated ==0:
		return 0
	## sigma(x)
	person1_sum = sum([dataset[person1][name] for idx,(name,rank) in enumerate(both_rated.items())])
	person2_sum = sum([dataset[person2][name] for idx,(name,rank) in enumerate(both_rated.items())])
	## sigma(x^2)
	person1_squared_sum = sum([pow(dataset[person1][name],2) for idx,(name,rank) in enumerate(both_rated.items())])
	person2_squared_sum = sum([pow(dataset[person2][name],2) for idx,(name,rank) in enumerate(both_rated.items())])
	## sigma(xy)
	product_sum_of_both_user = sum([dataset[person1][name] * dataset[person2][name] for idx,(name,rank) in enumerate(both_rated.items())])
	## computing the pearson correlation
	numerator = product_sum_of_both_user - (person1_sum * person2_sum / number_of_rated)
	denominator = sqrt((person1_squared_sum - pow(person1_sum,2) / number_of_rated) * (person2_squared_sum - pow(person2_sum,2) / number_of_rated) )
	if denominator ==0:
		return 0
	pearson_score = numerator / float(denominator)
	return pearson_score

def get_most_similar_user(user,number_of_users = 3):
	return sorted([(perason_correlation(user,others),others) for others in dataset if others != user],reverse=True)

def user_recommendation(person):
	total = {}
	similarity_sum = {}
	ranking_list = []

	for other in dataset:
		if other != person:
			pearson_score = perason_correlation(person,other)
			if pearson_score <=0:
				continue
			for item in dataset[other]:
				if item not in dataset[person] or dataset[person][item] ==0:
					## similarity * score(ranking)
					total.setdefault(item,0)
					total[item] += pearson_score * dataset[other][item]
					similarity_sum.setdefault(item,0)
					similarity_sum[item] += pearson_score
	ranking = sorted([(score/similarity_sum[name],name) for name,score in total.items()],reverse=True)
	recommendation_list = [name for score,name in ranking]
	return recommendation_list
### Euclidean Distance
def similarity(person1,person2):

	## find the items which are viewed by both person1 and person2
	both_rated = {}
	for idx,(name,rank) in enumerate(dataset[person1].items()):
		if name in dataset[person2]:
			both_rated[name] =1

	if len(both_rated) ==0:
		return 0
	## computing euclidean distance
	print('Computing Euclidean Distance ....')
	euclidean_distance =  []
	for idx,(name,rank) in enumerate(dataset[person1].items()):
		if name in dataset[person2]:
			euclidean_distance.append(pow(dataset[person1][name] - dataset[person2][name],2))
	sum_euclidean_distance = sum(euclidean_distance)
	return 1/sqrt(sum_euclidean_distance)


if __name__ == '__main__':
	print('Euclidean Distance : {} '.format(similarity('Lisa Rose','Jack Matthews')))
	print('Pearson Correlation : {}'.format(perason_correlation('Lisa Rose','Gene Seymour')))
	most_similar_user = get_most_similar_user('Lisa Rose',3)
	print('Most Similar users : ')
	print(most_similar_user)
	print('Recommendation for Toby : ')
	print(user_recommendation('Toby'))

