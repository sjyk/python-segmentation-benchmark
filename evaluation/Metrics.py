"""
This module implements metrics for 
the evaluation algorithms.
"""
import numpy as np

def nearest_neighbor_match(seq1, seq2):

	intersection_list = []

	for s1 in seq1:

		mine = None
		minv = None

		for s2 in seq2: 
			if minv == None or abs(s1-s2) < minv:
				minv = abs(s1-s2)
				mine = s2

		intersection_list.append((s1,mine))

	return intersection_list

def jaccard(seq1, seq2, tol=5):
	seq1 = seq1[1:len(seq1)-1]
	seq2 = seq2[1:len(seq2)-1]

	if len(seq1) == 0 or len(seq2) == 0:
		return 0

	intersection_list = nearest_neighbor_match(seq1, seq2)
	inter = len([t for t in intersection_list if abs(t[0]-t[1]) <= tol])
	union = len(seq2) + len(seq1) - inter
	return float(inter)/union
