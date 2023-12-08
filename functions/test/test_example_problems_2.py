import unittest
import numpy as np
import sys
sys.path.insert(1,"../")

from functions.example_problems_2 import ALOSDim, ScalingExpSine, MixedSine, ShortColumn, ShortColumn1U, UncertainEllipse

class ProblemDiffTest2(unittest.TestCase):

    
    def test_ALOSDim1Gradient(self):
        h = 1e-5
        dim = 1
        trueFunc = ALOSDim(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_ALOSDim2Gradient(self):
        h = 1e-5
        dim = 2
        trueFunc = ALOSDim(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_ALOSDim3Gradient(self):
        h = 1e-5
        dim = 3
        trueFunc = ALOSDim(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_ShortColumnGradient(self):
        h = 1e-5
        dim = 5
        trueFunc = ShortColumn(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_ShortColumn1UGradient(self):
        h = 1e-5
        dim = 3
        trueFunc = ShortColumn1U(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)


    def test_ScalingExpSine3Gradient(self):
        h = 1e-5
        dim = 3
        trueFunc = ScalingExpSine(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_ScalingExpSine12Gradient(self):
        h = 1e-5
        dim = 12
        trueFunc = ScalingExpSine(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_MixedSineGradient(self):
        h = 1e-5
        dim = 2
        trueFunc = MixedSine(ndim=dim)
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

    def test_UncertainEllipseGradient(self):
        h = 1e-5
        trueFunc = UncertainEllipse(fix_radius=None, fix_foci=None, fix_loc=None)
        dim = trueFunc.options["ndim"]
        xi = np.random.rand(dim)
        spread = trueFunc.xlimits[:,1] - trueFunc.xlimits[:,0]
        offset = -trueFunc.xlimits[:,0]
        xg = np.random.rand(dim)*spread - offset

        fgm = trueFunc(np.array([xg-h*xi]))
        fgp = trueFunc(np.array([xg+h*xi]))
        ga = np.zeros([1,dim])
        gat = np.zeros(dim)
        for i in range(dim):
            ga[0,i] = trueFunc(np.array([xg]), i)
            xg[i] += h
            gat[i] = trueFunc(np.array([xg]))
            xg[i] -= 2*h
            gat[i] -= trueFunc(np.array([xg]))
            gat[i] /= 2*h
            xg[i] += h

        finitediff = (1./(2*h))*(fgp-fgm)
        analytic = np.dot(ga, xi)
        err = abs(analytic - finitediff)
        self.assertTrue(err < 1.e-8)

if __name__ == '__main__':
    unittest.main()