#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn_crfsuite import CRF
from .utils import sent2features

# Conditional Random Field Model
class CRF_Model(object):
    
    # Initialize Parameters:
    # algorithm: Training algorithm, allows: 
    #   'lbfgs':    Gradient descent using the L-BFGS method
    #   'l2sgd':    Stochastic Gradient Descent with L2 regularization term
    #   'ap':       Averaged Perceptron
    #   'pa':       Passive Aggressive
    #   'arow':     Adaptive Regularization Of Weight Vector
    # c1: The coefficient for L1 Regularization 
    # c2: The coefficient for L2 Regularization 
    # max_iterations: The maximum number of iterations for optimization algorithms
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1,
                 max_iterations=100, all_possible_transitions=True):
        self.model = CRF(algorithm=algorithm,
                         c1=c1, c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    # Train the module with lists of sentences and tags
    def train(self, sent_lists, tag_lists):
        features = [sent2features(s) for s in sent_lists]
        self.model.fit(features, tag_lists)

    # Predict tags with lists of sentences
    def test(self, sent_lists):
        features = [sent2features(s) for s in sent_lists]
        tag_lists = self.model.predict(features)
        return tag_lists
