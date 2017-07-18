#!/usr/bin/env python

import numpy as np
import unittest
import time

from algorithms import BinaryGTD
from algorithms import GTD
from gvf import GVF

num_predictions = 1000
num_features = 1638
active_binary_features = 32


class testTime(unittest.TestCase):

    def test_time(self):
        print()

        #Test time for BinaryGTD
        theta = np.zeros((num_predictions,num_features))
        phi = np.random.randint(0, num_features, (active_binary_features))
        single_learners = [BinaryGTD(0.1, 0.01, 0.9, 0.9, np.copy(theta[i,:]), phi) for i in range(num_predictions)]
        multi_learner = BinaryGTD(0.1*np.ones((num_predictions))/active_binary_features, 0.01*np.ones((num_predictions))/active_binary_features, 0.9*np.ones((num_predictions)), 0.9*np.ones((num_predictions)), np.copy(theta), phi)
        base_learners = [GVF(alpha=0.1, 
                             beta=0.01,
                             num_features=len(phi),
                             off_policy=True, 
                             alg=GTD,
                             lambda_=lambda phi: np.random.random(),
                             gamma=lambda phi: np.random.random(),
                             cumulant=lambda phi: np.random.random(),
                             rho=lambda action, phi: np.random.random(),
                             policy=Policy(),
                             ) for _ in range(num_predictions)]


        start_time = time.time()
        #Test base class learners
        old_phi = phi
        for i in range(100):
            phi = np.random.randint(0, num_features, (active_binary_features))
            r = np.random.randint(-1, 1, (num_predictions))
            alpha = np.random.sample((num_predictions))/active_binary_features
            beta = np.random.sample((num_predictions))/active_binary_features
            for j in range(len(base_learners)):
                base_learners[j].alpha = alpha[j]
                base_learners[j].alphaH = beta[j]
                base_learners[j].update(0, phi)
        print("GVF Base Class time (single): ", time.time()-start_time)


        start_time = time.time()
        #Test single learners
        for i in range(100):
            phi = np.random.randint(0, num_features, (active_binary_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            rho = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))/active_binary_features
            beta = np.random.sample((num_predictions))/active_binary_features
            lambda_ = np.random.sample((num_predictions))
            for j in range(len(single_learners)):
                single_learners[j].update(phi, r[j], rho[j], alpha = alpha[j], beta = beta[j], lambda_ = lambda_[j], gamma = gamma[j])
        print("Binary single time: ", time.time()-start_time)

        start_time = time.time()
        #Test multiple learners
        for i in range(100):
            phi = np.random.randint(0, num_features, (active_binary_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            rho = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))/active_binary_features
            beta = np.random.sample((num_predictions))/active_binary_features
            lambda_ = np.random.sample((num_predictions))
            multi_learner.update(phi, r, rho, alpha = alpha, beta = beta, lambda_ = lambda_, gamma = gamma)
        print("Binary multiple time: ", time.time()-start_time)


        #Test time for GTD
        theta = np.zeros((num_predictions,num_features))
        phi = np.random.sample((num_features))
        single_learners = [GTD(0.1, 0.01, 0.9, 0.9, np.copy(theta[i,:]), phi) for i in range(num_predictions)]
        multi_learner = GTD(0.1*np.ones((num_predictions)), 0.01*np.ones((num_predictions)), 0.9*np.ones((num_predictions)), 0.9*np.ones((num_predictions)), np.copy(theta), phi)
        start_time = time.time()
        #Test single learners
        for i in range(100):
            phi = np.random.sample((num_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            rho = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))
            alpha /= np.sum(phi)
            beta = np.random.sample((num_predictions))
            beta /= np.sum(phi)
            lambda_ = np.random.sample((num_predictions))
            for j in range(len(single_learners)):
                single_learners[j].update(phi, r[j], rho[j], alpha = alpha[j], beta = beta[j], lambda_ = lambda_[j], gamma = gamma[j])
        print("Nonbinary single time: ", time.time()-start_time)

        start_time = time.time()
        #Test multiple learners
        for i in range(100):
            phi = np.random.sample((num_features))
            r = np.random.randint(-1, 1, (num_predictions))
            gamma = np.random.sample((num_predictions))
            rho = np.random.sample((num_predictions))
            alpha = np.random.sample((num_predictions))
            alpha/=np.sum(phi)
            beta = np.random.sample((num_predictions))
            beta/=np.sum(phi)
            lambda_ = np.random.sample((num_predictions))
            multi_learner.update(phi, r, rho, alpha = alpha, beta = beta, lambda_ = lambda_, gamma = gamma)
        print("Nonbinary multiple time: ", time.time()-start_time)

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(testTime)
    unittest.TextTestRunner(verbosity=2).run(suite)
