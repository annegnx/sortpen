import numpy as np
from numpy.testing import assert_equal

from sortpen.metrics import clustered_ratio_1D


class TestMetrics():

    def test_clustered_ratio_1D(self):
        truth = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        pred_1 = np.array([0, 1, 1, 2, 2, 2, 0, 0, 0])
        pred_2 = truth.copy()
        pred_3 = np.zeros_like(truth)
        pred_4 = np.arange(len(truth))
        pred_5 = np.array([1, 1, 1, 2, 2, 2, 4, 5, 6])

        result_1 = clustered_ratio_1D(pred_1, truth)
        result_2 = clustered_ratio_1D(pred_2, truth)
        result_3 = clustered_ratio_1D(pred_3, truth)
        result_4 = clustered_ratio_1D(pred_4, truth)
        result_5 = clustered_ratio_1D(pred_5, truth)

        truth_result_1 = {"intra_clusters_ratio": 2.5/3.,
                          "inter_clusters_ratio": 1, "rmse": np.round(np.sqrt(28)/np.sqrt(42), 16)}
        truth_result_2 = {"intra_clusters_ratio": 1.,
                          "inter_clusters_ratio": 1, "rmse": 0.}
        truth_result_3 = {"intra_clusters_ratio": 1.,
                          "inter_clusters_ratio": 0, "rmse": 1.}
        truth_result_4 = {"intra_clusters_ratio": 0.,
                          "inter_clusters_ratio": 0, "rmse": np.round((np.sqrt(66)/np.sqrt(42)), 16)}
        truth_result_5 = {"intra_clusters_ratio": 2./3.,
                          "inter_clusters_ratio": 2./3., "rmse": np.round(np.sqrt(14.)/np.sqrt(42), 16)}

        assert_equal(result_1, truth_result_1, "Test 1")
        assert_equal(result_2, truth_result_2, "Test 2")
        assert_equal(result_3, truth_result_3, "Test 3")
        assert_equal(result_4, truth_result_4, "Test 4")
        assert_equal(result_5, truth_result_5, "Test 5")


tester = TestMetrics()
tester.test_clustered_ratio_1D()
