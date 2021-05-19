#testing module
from preprocess import get_distance, find_center
import math 


def test_distance():
	'''test get_distance'''
	resp = get_distance([[0,0]], [[2,2]])
	assert resp == math.sqrt(2**2 + 2**2)


def test_distance_2():
	'''test get_distance'''
	resp = get_distance([[3,1]], [[10,20]])
	assert resp == math.sqrt(7**2 + 19**2)


def test_find_center():
	'''test find_center'''
	resp = find_center([[0,0]], [[2,2]])
	assert resp == [[1,1]]


def test_find_center():
	'''test find_center'''
	resp = find_center([[3,1]], [[10,20]])
	assert resp == [[3.5,9.5]]