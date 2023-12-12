import numpy as np
from infill.refinecriteria import looCV, HessianFit, TEAD
from infill.aniso_criteria import AnisotropicRefine
from infill.taylor_criteria import TaylorRefine, TaylorExploreRefine
from infill.hess_criteria import HessianRefine, HessianGradientRefine
from infill.loocv_criteria import POUSFCVT, SFCVT
from infill.aniso_transform import AnisotropicTransform


def GetCriteria(set, model0, gtrain0, xlimits, pdf_weight=None, sub_index=None):
    dim = xlimits.shape[0]
    fix_index = [k for k in np.arange(dim).tolist() if k not in sub_index]
    if(set.rtype == "aniso"):
        RC0 = AnisotropicRefine(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, rscale=set.rscale, nscale=set.nscale, neval=set.neval_fac*dim+set.neval_add, hessian=set.hess, interp=set.interp, bpen=set.bpen, objective=set.obj, multistart=set.multistart)
    elif(set.rtype == "anisotransform"):
        RC0 = AnisotropicTransform(model0, sequencer, gtrain0, sub_index=sub_index, pdf_weight=pdf_weight, nmatch=set.nmatch, neval=set.neval_fac*dim+set.neval_add, hessian=set.hess, interp=set.interp)
    elif(set.rtype == "tead"):
        RC0 = TEAD(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, gradexact=True)
    elif(set.rtype == "taylor"):
        RC0 = TaylorRefine(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, volume_weight=set.perturb, rscale=set.rscale, improve=set.pperb, multistart=set.multistart)
    elif(set.rtype == "taylorexp"):
        RC0 = TaylorExploreRefine(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, rscale=set.rscale, objective=set.obj, multistart=set.multistart)
    elif(set.rtype == "hess"):
        RC0 = HessianRefine(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, neval=set.neval_fac*dim+set.neval_add, rho=set.rho, rscale=set.rscale, multistart=set.multistart, scale_by_volume=set.vsca, return_rescaled=set.rsca, print_rc_plots=set.rc_print)
    elif(set.rtype == "hessgrad"):
        RC0 = HessianGradientRefine(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, neval=set.neval_fac*dim+set.neval_add, rho=set.rho, rscale=set.rscale, multistart=set.multistart, scale_by_volume=set.vsca, return_rescaled=set.rsca, grad_select=fix_index, print_rc_plots=set.rc_print)
    elif(set.rtype == "poussa"):
        RC0 = POUSSA(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print)
    elif(set.rtype == "pousfcvt"):
        RC0 = POUSFCVT(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight, improve=set.pperb, multistart=set.multistart, print_rc_plots=set.rc_print)
    elif(set.rtype == "sfcvt"):
        RC0 = SFCVT(model0, gtrain0, xlimits, sub_index=sub_index, pdf_weight=pdf_weight,  print_rc_plots=set.rc_print) # improve=pperb, multistart=multistart, not implemented
    else:
        raise ValueError("Given criteria not valid.")

    return RC0