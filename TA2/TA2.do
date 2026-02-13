********************************************************************************
* TA Session - Causal Inference & ML
* Barcelona School of Economics
* 
* This do file combines:
*   1. Non Linear panel data models
*   2. Nonparametric Density Estimation
*	3. Nonparametric Regression: Nadaraya-Watson & Local Linear
*	4. Semiparametric regression: Robinson Difference Estimator
*
* Author: Lucia Sauer
* Date: February 2026
********************************************************************************

********************************************************************************
* 1. Non Linear panel data models
********************************************************************************
{
* Clear workspace
clear all
set more off

* Load the CRIME dataset from Wooldridge
* County-level data, North Carolina, years 1981-1987
bcuse crime4, clear


* Set panel structure
* county = county identifier, year = 81 to 87
xtset county year


/*==============================================================================
  CREATE BINARY OUTCOME VARIABLE
  
  Since this is county-level data, we need to create a binary outcome
  Outcome: High crime rate (above median in each year)
==============================================================================*/

*High crime = more than 4 crimes per 100 residents per year
gen high_crime = (crmrte > 0.04)
label variable high_crime "Crime rate > 4 per 100 persons"
* Check distribution
tab high_crime year

* Check within-county variation (needed for FE)
bysort county: egen crime_sd = sd(high_crime)
count if crime_sd > 0 & !missing(crime_sd)
count if crime_sd == 0


/*==============================================================================
  Linear Probability Model with Fixed Effects
==============================================================================*/

* Research question: What determines whether a county has high crime?
* Predictors: arrest probability, conviction probability, police per capita,
*             population density, economic factors (wages)

* LPM with county fixed effects
xtreg high_crime lprbarr lprbconv lpolpc ldensity lwcon, fe vce(cluster county)
estimates store lpm_fe

* Check predicted values
predict p_lpm if e(sample)
summarize p_lpm

* How many predicted probabilities are outside [0,1]?
count if p_lpm < 0 & !missing(p_lpm)
count if p_lpm > 1 & !missing(p_lpm)
di "Percentage outside [0,1]: " ///
    (r(N) / _N) * 100 "%"


/*==============================================================================
  DISCUSSION
  
  Advantages of LPM:
  - Easy interpretation: coefficients are marginal effects
  - Consistent if strict exogeneity holds
  - Controls for county-specific time-invariant factors
  
  Disadvantages:
  - Predicted probabilities can be < 0 or > 1
  - Heteroskedasticity by construction
  - Constant marginal effects (unrealistic)
  - Linear relationship assumed (may not fit binary outcomes well)
==============================================================================*/


/*==============================================================================
  Nonlinear Models - Incidental Parameters Problem
==============================================================================*/

* Incidental Parameters Problem:
* - In principle, we would estimate 90 county FE
* - Each county has T=7 observations
* - As N→∞ but T is fixed, estimates would be INCONSISTENT in standard FE logit
* - Conditional logit solves this by conditioning on sufficient statistic
* - Bias of order O(1/T) ≈ 1/7 ≈ 14% is AVOIDED by using conditional likelihood

*Solution:
* * The conditional logit eliminates county FE by conditioning on sufficient statistic: Σ(t=1 to T) y_it for each county
* This allows consistent estimation of β without estimating α_i
* Only counties with variation (0 < Σy_it < T) contribute to likelihood

* Conditional logit reporting ODDS RATIOS
xtlogit high_crime lprbarr lprbconv lpolpc ldensity lwcon, fe or
estimates store clogit_fe

* Interpretation guide for Odds Ratios:
* OR = 1:   No effect
* OR > 1:   Variable INCREASES odds of high crime
* OR < 1:   Variable DECREASES odds of high crime


* Note how many observations/groups are used
di "Groups with variation: " e(N_g)
}

********************************************************************************
* 2. Nonparametric Density Estimation
********************************************************************************
{

clear all
set more off

sysuse auto, clear

*car price in dollars
sum price, detail


/*==============================================================================
  1. HISTOGRAMS - Different number of bins
==============================================================================*/

* Histogram with 10 bins
histogram price, bin(10) density ///
    title("Histogram: 10 bins") name(hist10, replace)

* Histogram with 20 bins
histogram price, bin(20) density ///
    title("Histogram: 20 bins") name(hist20, replace)

* Histogram with 50 bins
histogram price, bin(50) density ///
    title("Histogram: 50 bins") name(hist50, replace)

* Compare all three
graph combine hist10 hist20 hist50, ///
    rows(1) name(hist_compare, replace)
	
	
* First, check range of price to choose appropriate widths
sum price, detail
* Range is approximately 3,291 to 15,906 → range ≈ 12,615

* Histogram with wide bins (width = 2000)
histogram price, width(2000) density ///
    title("Histogram: width = 2000") name(hist_wide, replace)

* Histogram with medium bins (width = 1000)
histogram price, width(1000) density ///
    title("Histogram: width = 1000") name(hist_med, replace)

* Histogram with narrow bins (width = 500)
histogram price, width(500) density ///
    title("Histogram: width = 500") name(hist_narrow, replace)

* Compare all three
graph combine hist_wide hist_med hist_narrow, ///
    rows(1) name(hist_compare, replace)


/*==============================================================================
  2. KERNEL DENSITY - Different bandwidths
==============================================================================*/

/*==============================================================================
  Calculate Silverman's Plug-in Bandwidth Manually
==============================================================================*/

* Get statistics
quietly sum price, detail
local n = r(N)
local sd = r(sd)
local q1 = r(p25)
local q3 = r(p75)
local iqr = `q3' - `q1'

* Silverman's formula components
local sigma_robust = min(`sd', `iqr'/1.349)
di "Robust sigma: " `sigma_robust'

* For Epanechnikov kernel: delta = 1.7188 (from table)
local delta = 1.7188

* Silverman's plug-in bandwidth
local h_default = 1.364 * `delta' * `sigma_robust' * `n'^(-0.2)
di "Silverman's bandwidth (manual): " `h_default'

* Alternative formula (0.9 version)
local h_alt_default = 0.9 * `sigma_robust' * `n'^(-0.2)
di "Alternative bandwidth formula: " `h_alt_default'


/*==============================================================================
  Question 3: Three Bandwidths (default, half, double)
==============================================================================*/

* Get Stata's default bandwidth by running kdensity
kdensity price, kernel(epan) generate(x_def f_def) nograph
di "Stata's default bandwidth"

* Calculate bandwidth variations
local h_half = `h_alt_default' / 2
local h_double = `h_alt_default' * 2

di "Bandwidths to use:"
di "  h_default = " `h_alt_default'
di "  h_half = " `h_half'
di "  h_double = " `h_double'

* Generate KDE with three bandwidths
kdensity price, kernel(epan) width(`h_default') ///
    generate(x1 f1) nograph
    
kdensity price, kernel(epan) width(`h_half') ///
    generate(x2 f2) nograph
    
kdensity price, kernel(epan) width(`h_double') ///
    generate(x3 f3) nograph

* Plot all three
twoway (line f2 x2, lcolor(red) lwidth(medium)) ///
       (line f1 x1, lcolor(blue) lwidth(thick)) ///
       (line f3 x3, lcolor(green) lwidth(medium)), ///
       legend(order(1 "h/2 (undersmoothed)" ///
                    2 "h (plug-in)" ///
                    3 "2h (oversmoothed)")) ///
       title("Bias-Variance Trade-off") ///
       subtitle("Epanechnikov kernel") ///
       xtitle("Price") ytitle("Density") ///
       name(bandwidth_compare, replace)

/*
INTERPRETATION:
- h/2 (red): HIGH VARIANCE, LOW BIAS - wiggly, captures details but noisy
- h (blue): BALANCED - plug-in optimizes bias-variance trade-off
- 2h (green): LOW VARIANCE, HIGH BIAS - smooth but may miss features
*/


/*==============================================================================
  Question 4: Three Different Kernels (plug-in bandwidth)
==============================================================================*/

* Epanechnikov kernel (optimal)
kdensity price, kernel(epan) generate(x_epan f_epan) nograph

* Gaussian kernel
kdensity price, kernel(gauss) generate(x_gauss f_gauss) nograph

* Biweight kernel
kdensity price, kernel(bi) generate(x_bi f_bi) nograph

* Compare kernels
twoway (line f_epan x_epan, lcolor(blue) lwidth(medium)) ///
       (line f_gauss x_gauss, lcolor(red) lpattern(dash) lwidth(medium)) ///
       (line f_bi x_bi, lcolor(green) lpattern(dot) lwidth(medium)), ///
       legend(order(1 "Epanechnikov" 2 "Gaussian" 3 "Biweight")) ///
       title("Comparison of Kernel Functions") ///
       subtitle("All using plug-in bandwidth") ///
       xtitle("Price") ytitle("Density") ///
       name(kernel_compare, replace)

/*
INTERPRETATION:
- All three kernels produce VERY SIMILAR results
- Differences are minimal
- Confirms theory: kernel choice matters LESS than bandwidth choice
- Epanechnikov is theoretically optimal but gain is small
*/


/*==============================================================================
  Question 5: Compare with Normal Distribution
==============================================================================*/
* Kernel density (use default)
kdensity price, kernel(epan) normal 

}

********************************************************************************
*3. Nonparametric Regression: Nadaraya-Watson & Local Linear
********************************************************************************
{
sysuse nlsw88, clear

/*==============================================================================
  Part 1: Nadaraya-Watson (Local Constant)
==============================================================================*/

/*------------------------------------------------------------------------------
  What is NW doing?
  
  At each point x₀:
  1. Calculate weights K((xᵢ - x₀)/h) for all observations
  2. Weighted average: m̂(x₀) = Σ Kᵢ yᵢ / Σ Kᵢ
  
  Equivalent to fitting: min Σ[yᵢ - β₀]² · K((xᵢ - x₀)/h)
------------------------------------------------------------------------------*/

* Basic NW with default bandwidth
lpoly wage tenure, ///
    degree(0) ///
    kernel(epan) ci ///
    title("Nadaraya-Watson Estimator") ///
    subtitle("Local constant, degree = 0") ///
    xtitle("tenure") ytitle("wage") ///
    name(nw_default, replace)

/*
INTERPRETATION:
- Smooth curve through the data
- Each point = weighted average of nearby prices
- Bandwidth chosen automatically (Stata default)
*/


/*==============================================================================
  Part 2: Effect of Bandwidth (Bias-Variance Trade-off)
==============================================================================*/

* Calculate reasonable bandwidth range
quietly sum tenure
local range = r(max) - r(min)
local h_small = `range' / 40   // Undersmoothed
local h_medium = `range' / 20  // Balanced
local h_large = `range' / 5    // Oversmoothed

di "Bandwidths:"
di "  Small (undersmoothed): " `h_small'
di "  Medium (balanced): " `h_medium'
di "  Large (oversmoothed): " `h_large'

quietly summarize wage
local ymin = r(min)
local ymax = r(max)

* NW with small bandwidth (high variance, low bias)
lpoly wage tenure, degree(0) ci bwidth(`h_small') ///
    title("h = small") ///
    yscale(range(`ymin' `ymax')) ///
	ytitle("Hourly wage") ///
    legend(off) ///
    name(nw_small, replace)

* NW with medium bandwidth (balanced)
lpoly wage tenure, degree(0) ci bwidth(`h_medium') ///
    title("h = medium") ///
    yscale(range(`ymin' `ymax')) ///
	ytitle("") ///
    ylabel(, nolabel) ///
    legend(off) ///
    name(nw_medium, replace)

* NW with large bandwidth (low variance, high bias)
lpoly wage tenure, degree(0) ci bwidth(`h_large') ///
    title("h = large") ///
    yscale(range(`ymin' `ymax')) ///
	ytitle("") ///
    ylabel(, nolabel) ///
    legend(off) ///
    name(nw_large, replace)

* Compare all three
graph combine nw_small nw_medium nw_large, ///
    title("Bias-Variance Trade-off: Effect of Bandwidth") ///
    rows(1) ///
    ycommon ///
    imargin(tiny) ///
    name(bandwidth_comparison, replace)

/*
INTERPRETATION:
- Small h: wiggly, high variance, low bias (overfits)
- Medium h: smooth, balanced bias-variance
- Large h: very smooth, low variance, high bias (underfits)
Optimal h balances these extremes
*/




/*==============================================================================
  Part 3: Local Linear Regression
==============================================================================*/

/*------------------------------------------------------------------------------
  What is Local Linear doing?
  
  At each point x₀:
  Solve: min Σ[yᵢ - β₀ - β₁(xᵢ - x₀)]² · K((xᵢ - x₀)/h)
  
  Returns: m̂(x₀) = β̂₀
           m̂'(x₀) = β̂₁ (derivative!)
------------------------------------------------------------------------------*/

* Local linear (degree = 1, default)
lpoly wage tenure, ///
    degree(1) ///
    ci ///
    title("Local Linear Regression") ///
    subtitle("Degree = 1 (default)") ///
    xtitle("tenure") ytitle("wage") ///
    name(ll_default, replace)

/*
INTERPRETATION:
- Fits local LINE (not constant)
- Better at boundaries
- Adapts to local slope
*/


/*==============================================================================
  Part 4: Compare NW vs Local Linear
==============================================================================*/

* NW (degree 0)
lpoly wage tenure, ///
    degree(0) ///
    generate(x_nw m_nw) ///
    nograph

* Local linear (degree 1)
lpoly wage tenure, ///
    degree(1) ///
    generate(x_ll m_ll) ///
    nograph

* Plot both on same graph
twoway (line m_nw x_nw, lcolor(red) lwidth(medium)) ///
       (line m_ll x_ll, lcolor(blue) lwidth(medium)) ///
       (scatter wage tenure, mcolor(gray%30) msize(small)), ///
       legend(order(1 "NW (degree 0)" 2 "Local Linear (degree 1)" 3 "Data")) ///
       title("NW vs Local Linear") ///
       xtitle("tenure") ytitle("wage") ///
       name(nw_vs_ll, replace)

/*
INTERPRETATION:
- In the middle: very similar
- At boundaries: LL performs better (less bias)
- LL is default in most software for good reason
*/


/*==============================================================================
  Part 5: Using npregress (Modern Approach)
==============================================================================*/

* npregress kernel does local linear by default
npregress kernel wage tenure

/*
OUTPUT INTERPRETATION:
- Reports average marginal effect (derivative)
- More sophisticated than lpoly
- Can handle multiple covariates
*/

* Store results
estimates store np_ll

* Plot with confidence intervals
npgraph, ///
    title("Local Linear using npregress") ///
    name(npregress_plot, replace)

/*
ADVANTAGES of npregress:
1. Reports marginal effects directly
2. Better inference (robust SE)
3. Can control for other variables
4. More options (different kernels, bandwidths)
*/

*Takes some time, but usefull to see that you can estimate cluster bootstraped SE
*npregress kernel wage tenure, vce(bootstrap, reps(200))
*npgraph, ci title("Nonparametric fit with 95% CI")
*margins, at(tenure=(5 10 15 20)) dydx(tenure)

/*==============================================================================
  Part 6: Marginal Effects (Derivative Estimation)
==============================================================================*/

* Local linear gives derivative m'(x)
npregress kernel wage tenure

* Get marginal effect at different points
margins, at(tenure=(5 10 20 30)) dydx(tenure) vsquish

/*
INTERPRETATION:
- Shows how marginal effect of weight varies
- Example: if dy/dx = 0.186 at tenure = 10, then around tenure = 10,
  an additional 1 year of tenure is associated with about +0.186 in hourly wage
  (units: wage per year of tenure).
- Can see if returns are constant or diminishing
*/

	
}


********************************************************************************
*4. Robinson Difference Estimator
********************************************************************************
{
*ssc install semipar
bcuse hprice3, clear

semipar lprice larea lland rooms bath age if y81==1, ///
    nonpar(ldist) robust ci ///
    title("ldist non parametric") ///
    xtitle("ldist") ///
    ytitle("{&lambda}(ldist)")
}
