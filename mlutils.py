import operator
def majorityVote(label_list):
    label_count = {}
    for label in label_list:
        if label not in label_count.keys(): label_count[label] = 0
        label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]