"""Reproducible high-precision roots of the exact diagonal certificates."""
from decimal import Decimal, localcontext
from pathlib import Path
import json

N=[4,40,35,6,-29,-30,1,4,1]
D=[0,16,16,-6,-20,-16,4,6]
TRANSFORMED=[-64,-1024,-7496,-32384,-88654,-149492,-114888,
             110552,450072,654680,571208,319232,111872,22528,2048]

def polynomial(coefficients,x):
    value=Decimal(0)
    for c in coefficients[::-1]:value=value*x+c
    return value

def derivative(coefficients):
    return [i*c for i,c in enumerate(coefficients)][1:]

def bisect(function,low,high):
    for _ in range(400):
        mid=(low+high)/2
        if function(mid)>0:high=mid
        else:low=mid
    return (low+high)/2

def run(root):
    with localcontext() as ctx:
        ctx.prec=90
        tmin=bisect(lambda t:polynomial(derivative(N),t)*polynomial(D,t)
                    -polynomial(N,t)*polynomial(derivative(D),t),Decimal('.01'),Decimal('.99'))
        sigma_min=(-1/(2*tmin.ln())).sqrt()
        minimum=polynomial(N,tmin)/polynomial(D,tmin)
        def boundary(t):
            a=2*t/(1+t)**2
            return 3*t*t-(1-a)*(3-2*a)
        tc=bisect(boundary,Decimal('.01'),Decimal('.99'))
        u=tc*tc;a=2*tc/(1+tc)**2
        sigma_c=(-1/(2*tc.ln())).sqrt()
        r=4*a/(3-2*a);forcing=r*(1+u/4)+a*a
        kappa=3*forcing/(u*(r+3))
        reference_coefficient=(u*r/8+1-2*a+a*a/2)/forcing
        assert abs(float(sigma_min)-.7436676051037246)<2e-15
        assert abs(float(sigma_c)-.995394522106729)<2e-15
        values=dict(precision_decimal_digits=90,minimum_t=str(tmin),
                    minimum_sigma=str(sigma_min),minimum_expected_trials=str(minimum),
                    crossover_t=str(tc),crossover_sigma=str(sigma_c),crossover_u=str(u),
                    critical_survival_coefficient=str(kappa),
                    critical_reference_fraction_coefficient=str(reference_coefficient),
                    derivative_transformed_ascending_coefficients=TRANSFORMED)
    target=Path(root)/'results/finite/critical_constants.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(values,indent=2)+'\n')
    return values

if __name__=='__main__':run(Path(__file__).resolve().parents[1])
