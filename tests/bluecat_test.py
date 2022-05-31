import unittest
import numpy as np
from bluecat import EstimationTools, EmpiricalEstimation, KMomentsEstimation, Bluecat
from bluecat import NotNumpyError, SigLevelNotInRangeError, NoObservedDataError

qsim = np.array([50,30,80,20,87])
qcalib = np.array([96,  73, 188,  72,  82,  57,  28,  77, 116, 139,   0,  67,  91,
        52, 106,  80,  78,  71,  37,  22,  29, 113, 164,  62,  20,  39,
        58, 110, 170,  64, 193,  46,  87, 103, 138,  89, 136, 180,  75,
        21, 131, 174, 197, 196, 114, 142,  84, 122,  24, 185, 186, 162,
       120, 147, 146,  85,  61, 132, 129,  48, 199, 172, 155,  66, 192,
       118, 140, 191, 130, 126,  76,  79,  53,  41, 175,  65, 104, 190,
        90, 109,  88,  40, 112, 156, 145,   6,   5,  18, 181,  19,  31,
       183, 173, 184,  32,  69,  81,  83,  86,  13])
qcalibobs = qcalib-2
m = 20
siglev = 0.05
qsim_list = list(qsim)
qcalib_list = list(qcalib)
qcalibobs_list = list(qcalibobs)

class TestEstimationTools(unittest.TestCase):
    
    def test_num_of_m_neighbours_qossc(self):
        rc = EstimationTools.num_of_m_neighbours(qsim, qcalib, qcalibobs, m)
        np.testing.assert_array_equal(rc.qossc, np.array([ -2,   3,   4,  11,  16,  17,  18,  19,  20,  22,  26,  27,  29,
        30,  35,  37,  38,  39,  44,  46,  50,  51,  55,  56,  59,  60,
        62,  63,  64,  65,  67,  69,  70,  71,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  94,
       101, 102, 104, 107, 108, 110, 111, 112, 114, 116, 118, 120, 124,
       127, 128, 129, 130, 134, 136, 137, 138, 140, 143, 144, 145, 153,
       154, 160, 162, 168, 170, 171, 172, 173, 178, 179, 181, 182, 183,
       184, 186, 188, 189, 190, 191, 194, 195, 197]))
    
    def test_num_of_m_neighbours_vectmin(self):
        rc = EstimationTools.num_of_m_neighbours(qsim, qcalib, qcalibobs, m)
        np.testing.assert_array_equal(rc.vectmin, np.array([1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
       1.    , 1.    , 1.    , 1.    , 0.9875, 0.975 , 0.9625, 0.95  ,
       0.9375, 0.925 , 0.9125, 0.9   , 0.8875, 0.875 , 0.8625, 0.85  ,
       0.8375, 0.825 , 0.8125, 0.8   , 0.7875, 0.775 , 0.7625, 0.75  ,
       0.7375, 0.725 , 0.7125, 0.7   , 0.6875, 0.675 , 0.6625, 0.65  ,
       0.6375, 0.625 , 0.6125, 0.6   , 0.5875, 0.575 , 0.5625, 0.55  ,
       0.5375, 0.525 , 0.5125, 0.5   ]))
    
    def test_num_of_m_neighbours_vectmin1(self):
        rc = EstimationTools.num_of_m_neighbours(qsim, qcalib, qcalibobs, m)
        np.testing.assert_array_equal(rc.vectmin1, np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
       20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
       20, 19, 18, 17, 16, 14, 13, 11, 10,  8,  7,  5,  3,  1,  0]))
    
    def test_find_nearest(self):
        sortcalibsim = np.sort(qcalib)
        sortsim = np.sort(qsim)
        sim = sortsim[0]
        indatasimcal = EstimationTools.find_nearest(sortcalibsim, sim)
        self.assertEqual(indatasimcal, 6)
    
    def test_m_start_ind(self):
        indatasimcal = 6
        vectmin1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 19, 18, 17, 16, 14, 13, 11, 10,  8,  7,  5,  3,  1,  0])
        aux2 = EstimationTools.m_start_ind(indatasimcal,vectmin1)
        self.assertEqual(aux2,0)
    
    def test_m_end_ind(self):
        indatasimcal = 6
        vectmin1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 19, 18, 17, 16, 14, 13, 11, 10,  8,  7,  5,  3,  1,  0])
        vectmin = np.array([1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 0.9875, 0.975 , 0.9625, 0.95  ,
        0.9375, 0.925 , 0.9125, 0.9   , 0.8875, 0.875 , 0.8625, 0.85  ,
        0.8375, 0.825 , 0.8125, 0.8   , 0.7875, 0.775 , 0.7625, 0.75  ,
        0.7375, 0.725 , 0.7125, 0.7   , 0.6875, 0.675 , 0.6625, 0.65  ,
        0.6375, 0.625 , 0.6125, 0.6   , 0.5875, 0.575 , 0.5625, 0.55  ,
        0.5375, 0.525 , 0.5125, 0.5   ])
        aux1 = EstimationTools.m_end_ind(indatasimcal,vectmin,vectmin1)
        self.assertEqual(aux1,13)
    
    def test_m_neighbours(self):
        sortcalibsim = np.sort(qcalib)
        sortsim = np.sort(qsim)
        sim = sortsim[0]
        vectmin1 = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 19, 18, 17, 16, 14, 13, 11, 10,  8,  7,  5,  3,  1,  0])
        vectmin = np.array([1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    , 1.    ,
        1.    , 1.    , 1.    , 1.    , 0.9875, 0.975 , 0.9625, 0.95  ,
        0.9375, 0.925 , 0.9125, 0.9   , 0.8875, 0.875 , 0.8625, 0.85  ,
        0.8375, 0.825 , 0.8125, 0.8   , 0.7875, 0.775 , 0.7625, 0.75  ,
        0.7375, 0.725 , 0.7125, 0.7   , 0.6875, 0.675 , 0.6625, 0.65  ,
        0.6375, 0.625 , 0.6125, 0.6   , 0.5875, 0.575 , 0.5625, 0.55  ,
        0.5375, 0.525 , 0.5125, 0.5   ])
        aux2, aux1 = EstimationTools.m_neighbours(sortcalibsim, sim, vectmin, vectmin1)
        self.assertEqual(aux2, 0)
        self.assertEqual(aux1, 13)
    

class TestEmpiricalEstimation(unittest.TestCase):

    def test_empirical_uncertainty_estimation(self):
        rc = EmpiricalEstimation().uncertainty_estimation(qsim, qcalib, qcalibobs, m, siglev)
        np.testing.assert_almost_equal(rc.medpred, np.array([45.68292683, 30.0, 78.87804878, 16.15384615, 90.34146341]))
        np.testing.assert_almost_equal(rc.suppred, np.array([78,59,111,29,127]))
        np.testing.assert_almost_equal(rc.infpred, np.array([3,-2,50,-2,63]))


class TestKMomentsEstimation(unittest.TestCase):

    def test_k_moments_estimation(self):
        nstep = qsim.shape[0]
        medpred = np.array([45.68293,30.00000,78.87805,16.15385,90.34146])
        ptot, kp, kptail = KMomentsEstimation().k_moments_estimation(medpred,m,nstep)
        kp_test = [52.211258, 51.67516781076519, 54.40437196921146, 57.150961307524895, 
        59.89589832542405, 62.61860295217956, 65.29750199238656, 67.91080363987922, 
        70.43751726353949, 70.11017704932132, 73.07694662054095, 75.90903542011773, 
        78.56223910094151, 80.9921877732042, 79.04264723153587, 82.25696909544031, 
        85.14531331260501, 87.57640712689506, 76.91129155681458, 83.35633371093228, 90.34146]
        kptail_test = [52.211258, 34.882069757738975, 35.433785142641504, 35.76771571371742,
        35.854611436203676, 35.66820509634728, 35.18757254058687, 34.40005476629697,
        33.30472920699758, 24.68968277446495, 24.79509070898394, 24.691020798251337,
        24.355809526825624, 23.774547267649496, 18.828837774930914, 19.10998218091079,
        19.188260824985292, 19.007135752800217, 13.752417407412377, 14.904847799851177, 16.15385]
        np.testing.assert_allclose(np.array(kp), np.array(kp_test))
        np.testing.assert_allclose(np.array(kptail), np.array(kptail_test))

    def test_PBF_obj(self):
        nstep = 5
        m1 = 20
        m2 = np.arange(0,m1+1)
        ptot=nstep**(m2/m1)
        x = [0.001, 0.552, 20.0, 12.671]
        kp = [52.211258, 51.67516781076519, 54.40437196921146, 57.150961307524895, 
        59.89589832542405, 62.61860295217956, 65.29750199238656, 67.91080363987922, 
        70.43751726353949, 70.11017704932132, 73.07694662054095, 75.90903542011773, 
        78.56223910094151, 80.9921877732042, 79.04264723153587, 82.25696909544031, 
        85.14531331260501, 87.57640712689506, 76.91129155681458, 83.35633371093228, 90.34146]
        kptail = [52.211258, 34.882069757738975, 35.433785142641504, 35.76771571371742,
        35.854611436203676, 35.66820509634728, 35.18757254058687, 34.40005476629697,
        33.30472920699758, 24.68968277446495, 24.79509070898394, 24.691020798251337,
        24.355809526825624, 23.774547267649496, 18.828837774930914, 19.10998218091079,
        19.188260824985292, 19.007135752800217, 13.752417407412377, 14.904847799851177, 16.15385]
        lsquares = KMomentsEstimation().PBF_obj(x,ptot,kp,kptail)
        self.assertAlmostEqual(lsquares, 1.5415, places = 4)
    
    def test_fit_PBF(self):
        nstep = 5
        m1 = 20
        m2 = np.arange(0,m1+1)
        ptot=nstep**(m2/m1)
        kp = [52.211258, 51.67516781076519, 54.40437196921146, 57.150961307524895, 
        59.89589832542405, 62.61860295217956, 65.29750199238656, 67.91080363987922, 
        70.43751726353949, 70.11017704932132, 73.07694662054095, 75.90903542011773, 
        78.56223910094151, 80.9921877732042, 79.04264723153587, 82.25696909544031, 
        85.14531331260501, 87.57640712689506, 76.91129155681458, 83.35633371093228, 90.34146]
        kptail = [52.211258, 34.882069757738975, 35.433785142641504, 35.76771571371742,
        35.854611436203676, 35.66820509634728, 35.18757254058687, 34.40005476629697,
        33.30472920699758, 24.68968277446495, 24.79509070898394, 24.691020798251337,
        24.355809526825624, 23.774547267649496, 18.828837774930914, 19.10998218091079,
        19.188260824985292, 19.007135752800217, 13.752417407412377, 14.904847799851177, 16.15385]
        kp = np.array(kp)
        kptail = np.array(kptail)
        upparamd = [0.999, 5, 20, 15]
        lowparamd = [0.001,0.01,0.001,0]
        paramd = [sum(value)/2 for value in zip(upparamd,lowparamd)]
        x, opt = KMomentsEstimation().fit_PBF(paramd, lowparamd, upparamd, ptot, kp, kptail)
        #np.testing.assert_allclose(np.array(x),np.array([0.001, 0.552, 20, 12.672]),rtol=1e-3)
        self.assertAlmostEqual(opt.fun, 1.5415, places = 4)

    def test_estimate_order_p(self):
        x = [0.001, 0.552, 20, 12.672]
        ph, pl = KMomentsEstimation().estimate_order_p(x, siglev)
        self.assertAlmostEqual(ph, 21.30, places = 2)
        self.assertAlmostEqual(pl, 52.68, places = 2)

    def test_k_moments_uncertainty_estimation(self):
        rc = KMomentsEstimation().uncertainty_estimation(qsim, qcalib, qcalibobs, m, siglev)
        np.testing.assert_allclose(rc.medpred, np.array([45.68292683, 30.0, 78.87804878, 16.15384615, 90.34146341]),rtol=1e-3)
        np.testing.assert_allclose(rc.suppred, np.array([78.113549,  58.532715, 110.960679,   0.      , 126.287118]),rtol=1e-1)
        np.testing.assert_allclose(rc.infpred, np.array([0., 0., 0., 0., 0.]),rtol=1e-1)


class TestBluecat(unittest.TestCase):
    
    def test_bluecat_empirical(self):
        app = Bluecat(qsim, qcalib, qcalibobs, m, siglev, EmpiricalEstimation())
        app.sim()
        np.testing.assert_allclose(app.medpred, np.array([45.68292683, 30.0, 78.87804878, 16.15384615, 90.34146341]),rtol=1e-3)
        np.testing.assert_allclose(app.suppred, np.array([78,59,111,29,127]),rtol=1e-3)
        np.testing.assert_allclose(app.infpred, np.array([3,-2,50,-2,63]),rtol=1e-3)

    def test_bluecat_k_moments(self):
        app = Bluecat(qsim, qcalib, qcalibobs, m, siglev, KMomentsEstimation())
        app.sim()
        np.testing.assert_allclose(app.medpred, np.array([45.68292683, 30.0, 78.87804878, 16.15384615, 90.34146341]),rtol=1e-1)
        np.testing.assert_allclose(app.suppred, np.array([78.113549,  58.532715, 110.960679,   0.      , 126.287118]),rtol=1e-1)
        np.testing.assert_allclose(app.infpred, np.array([0., 0., 0., 0., 0.]),rtol=1e-1)
    
    def test_not_numpy_error_qsim(self):
        with self.assertRaises(NotNumpyError):
            Bluecat(qsim_list, qcalib, qcalibobs, m, siglev, EmpiricalEstimation())
        
    def test_not_numpy_error_qcalib(self):
        with self.assertRaises(NotNumpyError):
            Bluecat(qsim, qcalib_list, qcalibobs, m, siglev, EmpiricalEstimation())
    
    def test_not_numpy_error_qcalibobs(self):
        with self.assertRaises(NotNumpyError):
            Bluecat(qsim, qcalib, qcalibobs_list, m, siglev, EmpiricalEstimation())
    
    def test_sig_level_not_in_range(self):
        with self.assertRaises(SigLevelNotInRangeError):
            Bluecat(qsim, qcalib, qcalibobs, m, 1.2, EmpiricalEstimation())
    
    def test_no_observed_data(self):
        with self.assertRaises(NoObservedDataError):
            Bluecat(qsim, qcalib, qcalibobs, m, siglev, EmpiricalEstimation(),qobs=None,prob_plot=True)
    
    def test_ppoints(self):
        points = Bluecat.ppoints(np.array([1,2,3,4,5,6,7,8,9,10]))
        res = np.array([0.06097561, 0.15853659, 0.25609756, 0.35365854, 0.45121951,
        0.54878049, 0.64634146, 0.74390244, 0.84146341, 0.93902439])
        np.testing.assert_allclose(points, res, rtol=1e-1)



if __name__ == '__main__':
    unittest.main()
