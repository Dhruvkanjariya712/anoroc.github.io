import numpy
import pandas
import collections
import matplotlib as plt
from subprocess import check_output


class FungusClassifier(object):
    """Infers a hidden variable and uses Bayesian classification to predict whether a fungus is 
    edible or poisonous"""

    def __init__(self, filename):
        data = pandas.read_csv(filename, index_col=False)
        clusters = []
        for (i, row) in data.iterrows():
            best = -1
            sim = 0.5
            for (j, cluster) in enumerate(clusters):
                x = sum(cluster[key][value]/sum(cluster[key].values())
                        for (key, value) in row.items())/(data.shape[1])
                if x > sim:
                    best = j
                    sim = x
            if best == -1:
                clusters.append(collections.defaultdict(
                    lambda: collections.defaultdict(float)))
                print(i+1, 'rows analysed', len(clusters), 'clusters found')
            for (key, value) in row.items():
                clusters[best][key][value] += 1.0
        index = []
        for column in data.columns:
            index.extend([(column, value) for value in data[column].unique()])
        self.probabilities = pandas.DataFrame({(key, value): [cluster[key][value]+1.0 for cluster in clusters]
                                              for (key, value) in index}).T
        self.prior = self.probabilities.sum(axis=0)
        self.prior /= self.prior.sum()
        self.edibility_prior = self.probabilities.loc['class'].sum(axis=1)
        self.edibility_prior /= self.edibility_prior.sum()

        def normalize(group):
            return group.div(group.sum(axis=0), axis='columns')
        self.probabilities = self.probabilities.groupby(
            axis=0, level=0).apply(normalize)

    def __call__(self, **kwargs):
        "Estimates the probability that a fungus is edible given the features in kwargs"
        category = self.prior.copy()
        for (key, value) in kwargs.items():
            category *= self.probabilities.loc[(key, value)]
            category /= category.sum()
        result = self.edibility_prior * \
            ((self.probabilities.loc['class']*category).sum(axis=1))
        return result/result.sum()

    def test(self, filename):
        """Produces KDE plots of the estimated probability"""
        data = pandas.read_csv(filename, index_col=False)
        observables = [column for column in data.columns if column != 'class']
        results = pandas.DataFrame(
            [self(**row) for (i, row) in data[observables].iterrows()])
        results.loc[:, 'class'] = data['class']
        return results
