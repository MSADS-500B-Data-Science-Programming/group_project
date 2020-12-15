# European Bank Marketing Group Project
MSADS500B - Data Science Programming Group Project

### Abstract

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. The dataset is ordered by date (from May 2008 to November 2010).

While increasing the duration of the contact attempt in the campaign can stand to benefit a higher subscription rate, it will only do so slightly. Therefore, a revision model based on logistical regression is carried out to solve the overarching dilemma of targeted marketing to a select demographic, with 87% accuracy.

    Optimization terminated successfully.
             Current function value: 0.479469
             Iterations 7
                               Results: Logit
    ====================================================================
    Model:               Logit             Pseudo R-squared:  0.308     
    Dependent Variable:  deposit           AIC:               57647.1741
    Date:                2020-12-13 17:10  BIC:               57818.2380
    No. Observations:    60076             Log-Likelihood:    -28805.   
    Df Model:            18                LL-Null:           -41642.   
    Df Residuals:        60057             LLR p-value:       0.0000    
    Converged:           1.0000            Scale:             1.0000    
    No. Iterations:      7.0000                                         
    --------------------------------------------------------------------
                         Coef.  Std.Err.    z     P>|z|   [0.025  0.975]
    --------------------------------------------------------------------
    job_admin.          -1.1086   0.0447 -24.7750 0.0000 -1.1963 -1.0209
    job_blue-collar     -1.2743   0.0392 -32.4982 0.0000 -1.3511 -1.1974
    job_entrepreneur    -1.6783   0.1046 -16.0430 0.0000 -1.8833 -1.4732
    job_housemaid       -1.9375   0.1151 -16.8301 0.0000 -2.1632 -1.7119
    job_retired         -0.5353   0.0550  -9.7354 0.0000 -0.6430 -0.4275
    job_self-employed   -1.4575   0.0846 -17.2257 0.0000 -1.6233 -1.2916
    job_student         -0.9760   0.0805 -12.1232 0.0000 -1.1337 -0.8182
    job_unemployed      -1.3534   0.0863 -15.6813 0.0000 -1.5226 -1.1843
    job_unknown         -2.4720   0.1937 -12.7595 0.0000 -2.8517 -2.0923
    marital_divorced    -1.8412   0.0498 -36.9523 0.0000 -1.9388 -1.7435
    marital_married     -1.2229   0.0296 -41.3819 0.0000 -1.2809 -1.1650
    marital_single      -1.1418   0.0333 -34.3293 0.0000 -1.2070 -1.0767
    education_primary   -1.6621   0.0515 -32.2933 0.0000 -1.7630 -1.5612
    education_secondary -0.9882   0.0277 -35.6692 0.0000 -1.0426 -0.9339
    education_tertiary  -1.3234   0.0303 -43.7156 0.0000 -1.3828 -1.2641
    default_yes         -0.9444   0.1359  -6.9505 0.0000 -1.2107 -0.6781
    housing_no           0.1569   0.0292   5.3754 0.0000  0.0997  0.2141
    housing_yes         -0.7991   0.0313 -25.5400 0.0000 -0.8604 -0.7377
    loan_no              2.4685   0.0286  86.3215 0.0000  2.4124  2.5245
    ====================================================================

    
    
A recorded video of our presentation can be viewed here:

https://www.youtube.com/watch?v=-id0kLvWdJY