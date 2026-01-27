********************************************************************************
* Panel Data Analysis: Static and Dynamic Models
* TA Session - Causal Inference & ML
* Barcelona School of Economics
* 
* This do file combines:
*   1. Static Panel Data Models (Grunfeld data)
*   2. Dynamic Panel Data Models (Arellano-Bond with abdata)
*
* Author: Lucia Sauer
* Date: January 2026
********************************************************************************

clear all
set more off

* Set seed for reproducibility
set seed 12345


* Install required packages (uncomment if needed)
/*
ssc install estout, replace
ssc install xtoverid, replace
ssc install reghdfe, replace
ssc install ftools, replace
ssc install xttest3, replace
ssc install xtabond2, replace
ssc install outreg2, replace
*/


********************************************************************************
* PART I: STATIC PANEL DATA MODELS (GRUNFELD DATA)
********************************************************************************

webuse grunfeld, clear
describe

* Set Panel Structure
xtset company year

* Descriptive Statistics - Pooled summary statistics
summarize invest mvalue kstock

* Descriptive Statistics - Within and between variation
xtsum invest mvalue kstock

********************************************************************************
* Estimation Methods
********************************************************************************
{
*------------------------------------------------------------------------------
* Pooled OLS (ignoring panel structure)
*------------------------------------------------------------------------------

reg invest mvalue kstock, robust

*------------------------------------------------------------------------------
* Fixed Effects (Within Estimator)
*------------------------------------------------------------------------------

xtreg invest mvalue kstock, fe robust

* Store for later comparison
estimates store FE_robust

*------------------------------------------------------------------------------
* Random Effects (GLS Estimator)
*------------------------------------------------------------------------------

xtreg invest mvalue kstock, re robust

* Store for later comparison
estimates store RE_robust
}

********************************************************************************
* Fixed Effects vs Random Effects - Hausman Test
********************************************************************************

{
*------------------------------------------------------------------------------
* Standard Hausman Test (requires non-robust estimation)
*------------------------------------------------------------------------------

quietly xtreg invest mvalue kstock, fe
estimates store FE

quietly xtreg invest mvalue kstock, re
estimates store RE

hausman FE RE, sigmamore

*------------------------------------------------------------------------------
* Robust Hausman Test (Wooldridge's Approach)
*------------------------------------------------------------------------------

* Step 1: Create within-transformed (demeaned) variables
bysort company: egen mean_mvalue = mean(mvalue)
bysort company: egen mean_kstock = mean(kstock)

gen dm_mvalue = mvalue - mean_mvalue
gen dm_kstock = kstock - mean_kstock

* Step 2: Run RE regression with demeaned variables added
xtreg invest mvalue kstock dm_mvalue dm_kstock, re robust

* Step 3: Test if demeaned variables are jointly zero
test dm_mvalue dm_kstock

* Clean up
drop mean_mvalue mean_kstock dm_mvalue dm_kstock

*------------------------------------------------------------------------------
* Alternative: Using xtoverid
*------------------------------------------------------------------------------

xtreg invest mvalue kstock, re robust
xtoverid
}

********************************************************************************
* Clustered Standard Errors
********************************************************************************

{
*------------------------------------------------------------------------------
* Comparing standard errors: with and without clustering
*------------------------------------------------------------------------------

* Without clustering (INCORRECT for panel data)
quietly xtreg invest mvalue kstock, fe
estimates store FE_nocl

* With heteroskedasticity-robust only
quietly xtreg invest mvalue kstock, fe robust
estimates store FE_robust_only

* With cluster-robust (CORRECT - handles heterosk. AND serial correlation)
quietly xtreg invest mvalue kstock, fe vce(cluster company)
estimates store FE_cluster

* Compare all three
estimates table FE_nocl FE_robust_only FE_cluster, se stats(N) ///
    title("Comparing Standard Errors")

}

********************************************************************************
*Two-Way Fixed Effects
********************************************************************************

{
*------------------------------------------------------------------------------
* Entity and Time Fixed Effects
*------------------------------------------------------------------------------

xtreg invest mvalue kstock i.year, fe robust

*------------------------------------------------------------------------------
* Testing for Time Fixed Effects
*------------------------------------------------------------------------------

* Test if year dummies are jointly significant
testparm i.year


********************************************************************************
* Using reghdfe
********************************************************************************

* Entity fixed effects only

reghdfe invest mvalue kstock, absorb(company) vce(cluster company)

* Two-way fixed effects (entity and time)

reghdfe invest mvalue kstock, absorb(company year) vce(cluster company)
}

********************************************************************************
* Diagnostic Tests
********************************************************************************

{
* Testing for Heteroskedasticity
quietly xtreg invest mvalue kstock, fe
xttest3


*Testing for Serial Correlation
* Install xtserial if needed
capture which xtserial
if _rc != 0 {
    net from http://www.stata-journal.com/software/sj3-2/
	net describe st0039
	net install st0039
}
xtserial invest mvalue kstock
}

********************************************************************************
* Summary Comparison Table
********************************************************************************

{
* Pooled OLS with clustering
quietly reg invest mvalue kstock, vce(cluster company)
estimates store OLS

* Fixed Effects with clustering
quietly xtreg invest mvalue kstock, fe vce(cluster company)
estimates store FE_final

* Random Effects with clustering
quietly xtreg invest mvalue kstock, re vce(cluster company)
estimates store RE_final

* Two-Way Fixed Effects with clustering
quietly reghdfe invest mvalue kstock, absorb(company year) vce(cluster company)
estimates store TWFE

* Display comparison table
estimates table OLS FE_final RE_final TWFE, ///
    star stats(N r2 r2_w r2_b r2_o) ///
    title("Comparison of Panel Data Estimators")
}

********************************************************************************
* Export Results
********************************************************************************

{
*Export results to Latex using outreg2

quietly reg invest mvalue kstock, vce(cluster company)
outreg2 using panel_results.tex, replace ctitle(Pooled OLS) ///
    addtext(Firm FE, No, Year FE, No, Clustered SEs, Yes)

quietly xtreg invest mvalue kstock, fe vce(cluster company)
outreg2 using panel_results.tex, append ctitle(FE) ///
    addtext(Firm FE, Yes, Year FE, No, Clustered SEs, Yes)

quietly xtreg invest mvalue kstock, re vce(cluster company)
outreg2 using panel_results.tex, append ctitle(RE) ///
    addtext(Firm FE, --, Year FE, No, Clustered SEs, Yes)

quietly reghdfe invest mvalue kstock, absorb(company year) vce(cluster company)
outreg2 using panel_results.tex, append ctitle(Two-Way FE) ///
    addtext(Firm FE, Yes, Year FE, Yes, Clustered SEs, Yes)

}


********************************************************************************
* PART II: DYNAMIC PANEL DATA MODELS (ARELLANO-BOND)
********************************************************************************

webuse abdata, clear
describe 

xtset id year

********************************************************************************
* Naive Estimators
********************************************************************************

{
*Pooled OLS
reg n nL1 nL2 w wL1 k kL1 kL2 ys ysL1 ysL2 yr*, cluster(id)
estimates store dyn_ols
*Note: Coefficient on nL1 should be >1 due to upward bias from omitted c_i

*Random Effects
xtreg n nL1 nL2 w wL1 k kL1 kL2 ys ysL1 ysL2 yr*, re cluster(id)
estimates store dyn_re
*Note: Even with RE, nL1 is correlated with c_i → inconsistent

*Fixed Effects (Nickell Bias)
xtreg n nL1 nL2 w wL1 k kL1 kL2 ys ysL1 ysL2 yr*, fe cluster(id)
estimates store dyn_fe
*Note: Within transformation creates mechanical correlation
*Nickell bias ≈ -1/T, so coefficient on nL1 is biased downward
}

********************************************************************************
* Arellano-Bond GMM Estimator (Consistent)
********************************************************************************

{
xtabond2 n nL1 nL2 w wL1 k kL1 kL2 ys ysL1 ysL2 yr*, ///
gmm(L.(n w k), lag(2 .)) ///
iv(L(0/2).ys yr*) ///
noleveleq ///
robust ///
small
    
estimates store ab

*"Key diagnostics:
*- AR(2) test: Want p-value > 0.05 (no 2nd order serial correlation)
*- Hansen test: Want p-value > 0.05 (instruments are valid)
*- #instruments: Should be ≤ # groups


}

********************************************************************************
* Comparison of ALL Estimators
********************************************************************************

{
estimates table dyn_ols dyn_re dyn_fe ab, ///
    star stats(N ar2p hansenp) ///
    title("Dynamic Panel Models: Comparison of Estimators")


*Expected pattern: β̂_FE < β̂_AB < β̂_OLS/RE"
}

