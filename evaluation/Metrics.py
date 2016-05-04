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
	#remove the edge effects
	seq1 = [s for s in seq1 if abs(s) > 5 and abs(s-np.max(seq1)) > 5]
	#seq1[1:len(seq1)-1]
	seq2 = [s for s in seq2 if abs(s) > 5 and abs(s-np.max(seq2)) > 5]

	if len(seq1) == 0 or len(seq2) == 0:
		return 0

	intersection_list = nearest_neighbor_match(seq1, seq2)
	inter = len([t for t in intersection_list if abs(t[0]-t[1]) <= tol])
	union = len(seq2) + len(seq1) - inter
	return float(inter)/union


def segment_precision_recall (seg1,seg2):
    """
    s1: prediction -- should be [start, end]
    s2: ground truth subsegment- [start, end]
    """
    if len(seg1)>2 or len(seg2)>2:
        print "Incorrect input to segment_precision_recall"
        return

    print seg1, seg2

    len_ref = seg2[1] - seg2[0] #first point in the sequence not included
    len_pred = seg1[1] - seg1[0]
    int_start = max(seg1[0],seg2[0])
    int_end = min(seg1[1],seg2[1])

    #True positives = length of intersection
    TP = float(max( int_end - int_start  , 0 ))

    # recall = true_positive/ condition_positive
    if len_ref >0:        
        recall = TP/len_ref
    else:
        recall = 0
    # precision = true_positive / test_outcome_positive
    if len_pred>0:
        precision =  TP/len_pred
    else:
        precision =0

    return precision, recall

def inter_over_union (seg1,seg2):
    """
    s1: prediction -- should be [start, end]
    s2: ground truth subsegment- [start, end]
    """
    if len(seg1)>2 or len(seg2)>2:
        print "Incorrect input to inter_over_union"
        return

    inter_start = max(seg1[0],seg2[0])
    inter_end = min(seg1[1],seg2[1])
    inter_len = float(max( inter_end - inter_start  , 0 ))

    # Union = setA + setB - (setA intersect setB)
    union_len = (seg2[1] - seg2[0]) + (seg1[1] - seg1[0]) - inter_len

    return inter_len/union_len

def f1_score (seg1,seg2):
    """
    s1: prediction -- should be [start, end]
    s2: ground truth subsegment- [start, end]
    """
    if len(seg1)>2 or len(seg2)>2:
        print "Incorrect input to f1_score"
        return
    
    p, r = segment_precision_recall (seg1,seg2)

    # f1-score = harmonic mean of precision and recall
    return 2*p*r / (p+r)
    

def segment_correspondence (seq1,seq2, similarity_measure = "recall"):
    """
    seq1: prediction 
    seq2: ground truth subsegment
    """

    wt_mat = np.zeros((len(seq1)-1,len(seq2)-1))

    #populate the pairwise weights
    for i in range(len(seq1)-1):
        s1 = [seq1[i], seq1[i+1] ]

        for j in range(len(seq2)-1):
            s2 = [seq2[j], seq2[j+1] ]
            
            if similarity_measure == "recall":
                p, score = segment_precision_recall (s1,s2)

            elif similarity_measure == "f1_score":
                score = f1_score (s1,s2)                

            elif similarity_measure == "IOU":
                score = inter_over_union (s1,s2)                

            else:
                print "Need a valid similarity_measure for segment_correspondence"
                break     
            
            # print i,j, s1, s2 #debug            
            wt_mat[i,j] = score


    # association of each predicted segment with max wt in the row
    max_ind = np.argmax(wt_mat, axis=1)

    return max_ind, wt_mat


def frame_acc (seq1, seq2, similarity_measure='recall'):
    """
    Frame wise accuracy
    """
    max_ind, wt_mat = segment_correspondence (seq1,seq2, similarity_measure)
    acc_score = 0.0
    len_pred = seq1[-1] - seq1[0]

    for k1 in range(len(seq1)-1):
        for i in range (seq1[k1], seq1[k1+1]):
            start = seq2[max_ind[k1]]
            end = seq2 [max_ind[k1] + 1]
            
            print i, seq1[k1], max_ind[k1], start, end, acc_score
            if i >= start and i< end:
                #correctly matched
                acc_score = acc_score +1

    # return frame wise accuracy aggregated for the full sequence
    return acc_score/len_pred

def seg_acc (seq1, seq2, thresh = 0.4, similarity_measure = "recall"):
    """
    Segmentation Accuracy
    
    Defined as the ratio of the ground - truth segments that are correctly detected
    A GT segment is said to be detected if there exists a predicted segment with an overlap 
    greater than thresh. Overlap is defined as Inter-over-union

    Reference: http://watchnpatch.cs.cornell.edu/paper/watchnpatch_cvpr15.pdf (sec#6.3)    
    """
    #get correspondence
    max_ind, wt_mat = segment_correspondence (seq1,seq2, similarity_measure)

    num_gt_segments = len(seq2) - 1
    acc_score = np.zeros(num_gt_segments,)

    for k2 in range(num_gt_segments):
        for k1 in range(len(seq1)-1):

            s1 = [seq1[k1], seq1[k1+1]]
            s2 = [seq2[k2], seq2[k2+1]]            
            score = inter_over_union (s1, s2)
            
            #condition if the GT segment k2 is covered by any of predicted segments
            if score >= thresh:
                acc_score[k2] = 1.0

    #return ratio of GT segments covered
    return sum(acc_score)/num_gt_segments


def evaluate(seq1, seq2, method='jaccard', **options):
    """
    generic evaluation call
    seq1: predicted sequence -- algorithm output
     seq2: reference -- ground truth	
    method: which method to use. defaults to Jaccard (intersection over union)
    """
    if type == 'jaccard':
        if 'tol' in options.keys():
            return jaccard(seq1, seq2, tol=options['tol'])
        else:
            return jaccard(seq1, seq2)

    if method == 'frame_acc':
        if 'similarity_measure' in options.keys():
            return frame_acc(seq1, seq2, similarity_measure = options['similarity_measure'])
        else:
            return frame_acc(seq1, seq2)

    if method == 'seg_acc':

        if 'thresh' in options.keys():
                thresh = options['thresh']
        else:
                thresh = 0.4

        if 'similarity_measure' in options.keys():
            return frame_acc(seq1, seq2, similarity_measure = options['similarity_measure'], thresh = thresh)
        else:
            return seg_acc(seq1, seq2, thresh = thresh)

# ToDO: Implement this for later
def DTW(c1,c2,dist=lambda x,y:abs(x-y)):
    """
    Segmentwise DTW similarity
    """
    pass

def edit_distance(c1,c2,match=lambda x,y:x==y):
    """
    ToDO: implement edit distance based on segment level DTW
    """
    pass