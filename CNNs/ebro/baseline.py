from util import ReadCounters, ReadGold
import matplotlib.pyplot as plt



# generate the baseline results by using Lee's old method
def GenerateBaseline(prefix_one, prefix_two, threshold, maximum_distance):
    # get the ground truth
    ground_truth = ReadGold(prefix_one, prefix_two, threshold, maximum_distance)

    # read the counters
    _, _, _, scores = ReadCounters(prefix_one, prefix_two, threshold, maximum_distance)

    # prune the scores to only include valid ground truth
    pruned_scores = []
    pruned_ground_truth = []

    for iv in range(len(ground_truth)):
        if ground_truth[iv] == 2: continue

        pruned_scores.append(scores[iv])
        # subtract one so ground truth is true when it should be merged
        pruned_ground_truth.append(1 - ground_truth[iv])

    # find the number of positive and negative examples
    npositive_examples = 0
    nnegative_examples = 0
    for label in pruned_ground_truth:
        if label: npositive_examples += 1
        else: nnegative_examples += 1

    # keep track of the precision and recall at each step
    precisions = []
    recalls = []

    # tracker for positive and negative locations
    positives_seen = 0
    negatives_seen = 0

    # sort the ground truth in order of score
    for (score, ground_truth) in sorted(zip(pruned_scores, pruned_ground_truth), reverse=True):

        # increment the number of positve and negative examples seen
        if ground_truth: positives_seen += 1
        else: negatives_seen += 1

        # at this location, the recall is the number of true positives seen divided by the total number of positives
        recall = positives_seen / float(npositive_examples)
        # at this location, the precision is the number of true positives seen divided by the total number of examples seen
        precision = positives_seen / float(positives_seen + negatives_seen)

        precisions.append(precision)
        recalls.append(recall)

    # plot the precision and recall curve
    plt.plot(recalls, precisions, 'r')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.xlim([0,1.1])
    plt.ylim([0,1.1])
    plt.title('{} - {}'.format(prefix_one, prefix_two))
    plt.savefig('results/ebro/{}-{}-{}-{}nm-baseline.png'.format(prefix_one, prefix_two, threshold, maximum_distance))