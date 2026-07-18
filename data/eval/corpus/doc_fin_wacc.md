# Methodology for Calculating WACC for Privately Held Companies

**Document ID:** doc_fin_wacc.md
**Owner:** Valuation and Financial Analysis Group
**Last updated:** July 11, 2026

## Overview of the Weighted Average Cost of Capital

The **weighted average cost of capital** (WACC) represents the blended cost of a company's financing sources, reflecting the required return expected by both debt and equity providers. For **privately held** companies, the calculation of WACC presents unique challenges because market prices for the company's equity and debt are not directly observable. Despite these challenges, WACC remains a fundamental input for investment appraisal, business valuation, and financial reporting purposes.

## Estimating the Cost of Equity

The **cost of equity** for a privately held company is typically estimated using the **Capital Asset Pricing Model** (CAPM), which expresses the cost of equity as the risk-free rate plus a beta-adjusted equity risk premium. Since the company's own beta is unobservable, the standard approach is to use the **pure-play method**, which involves identifying a set of comparable publicly traded companies, calculating their leveraged betas, and unlevering them to remove the effect of their capital structures. The median or mean of these unlevered betas is then relevered using the private company's target capital structure to derive the appropriate beta for the company. The resulting cost of equity formula is: Re = Rf + Beta(levered) x ERP, where Rf is the risk-free rate (typically the yield on long-term US Treasury bonds) and ERP is the equity risk premium.

## Estimating the Cost of Debt

The **cost of debt** for a privately held company can be estimated based on the company's actual borrowing rate on its outstanding debt facilities. If the company has no debt or the existing debt is not representative, the cost of debt may be estimated by referencing the borrowing rates of comparable companies with similar credit profiles. The credit rating of the privately held company, if available, can be used to determine a synthetic credit spread to add to the risk-free rate. An alternative approach is to use the interest coverage ratio and other financial metrics to estimate a synthetic credit rating and corresponding default spread.

## The WACC Formula

The complete WACC formula for a privately held company is expressed as follows:

**WACC = (E/V x Re) + (D/V x Rd x (1 - T))**

where:
- E = market value of equity (estimated using the valuation exercise itself or a target capital structure)
- V = total market value of the firm (E + D)
- Re = cost of equity, estimated via the pure-play CAPM method
- D = market value of debt (typically approximated by book value for private companies)
- Rd = cost of debt, estimated from the company's borrowing rate or comparable company rates
- T = marginal corporate tax rate

## Practical Considerations and Limitations

Several practical considerations arise when applying the WACC methodology to a privately held company. First, the capital structure weights should be based on market values rather than book values, but market values are often unavailable, requiring the use of target capital structures or iterative valuation approaches. Second, the selection of comparable companies for the pure-play beta estimation requires careful judgment regarding industry classification, size, and business model similarity. Third, the cost of equity typically includes a size premium for small privately held companies, as empirical evidence suggests that smaller firms have higher expected returns. Meridian Analytics, as a privately held company, applies these adjustments when preparing valuations for internal decision-making and financial reporting purposes.
