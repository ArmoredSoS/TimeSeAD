import numpy

#accuracy, precision, recall, F1 score, CM
#threshold, trasforma array in 1-0

class Stats():
    def __init__(self, data, ground_truth, threshold: int = 0):
        self.threshold = threshold
        data = self.transform_data(data)
        self.true_positives = numpy.sum((data == 1) & (ground_truth == 1))
        self.true_negatives = numpy.sum((data == 0) & (ground_truth == 0))
        self.false_positives = numpy.sum((data == 1) & (ground_truth == 0))
        self.false_negatives = numpy.sum((data == 0) & (ground_truth == 1))
        self.total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives

        
    def transform_data(self, data):
        return numpy.array([1 if i > self.threshold else 0 for i in data], dtype=int)
        
    def accuracy(self):
        return (self.true_positives + self.true_negatives) / self.total
    
    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)
    
    def recall(self):
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    def F1score(self):
        return 2 * ((self.precision() * self.recall()) / (self.precision() * self.recall()))