# Data Sources for Labor Cost Analysis

This document explains the data sources used for labor cost estimation in the Bayesian pipeline.

## Overview

The system fetches **real economic data** to build Bayesian posteriors for labor costs, providing realistic uncertainty estimates for Monte Carlo simulations.

---

## United States 🇺🇸

**Source**: FRED (Federal Reserve Economic Data)  
**Indicator**: `CES3000000003` - Average Hourly Earnings, Manufacturing  
**Access**: ✅ **FREE** - No API key required  
**Frequency**: Monthly  
**URL Format**: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CES3000000003`

**Current Data**: ~$35.50/hour (as of Aug 2025)  
**Data Points**: 24 months of historical data

---

## Mexico 🇲🇽

**Source**: FRED (Federal Reserve Economic Data)  
**Indicator**: `LCEAMN01MXM661S` - Hourly Earnings Index (2015=100)  
**Access**: ✅ **FREE** - No API key required  
**Frequency**: Monthly  
**URL Format**: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=LCEAMN01MXM661S`

**Current Data**: Index ~219 (June 2025)  
**Data Points**: 24 months of historical data

---

## China 🇨🇳

**Source**: World Bank Open Data API  
**Indicator**: `NY.GDP.PCAP.PP.CD` - GDP per capita, PPP (current international $)  
**Access**: ✅ **FREE** - No API key required  
**Frequency**: Annual  
**URL Format**: `https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.PCAP.PP.CD?format=json`

**Why GDP per capita?**
- Direct wage data for China manufacturing is not freely available
- GDP per capita (PPP) is a strong proxy for wage levels
- Captures economic growth that translates to wage increases
- 15 years of historical data (2010-2024)

**Current Data**: $27,105 per capita (2024)  
**Growth**: 188% increase over 2010-2024 period

**Alternative Indicators** (also free from World Bank):
- `NY.GNP.PCAP.CD` - GNI per capita (Atlas method)
- Additional labor indicators can be explored at: https://data.worldbank.org/

---

## Data Processing Pipeline

### 1. Fetch Raw Data
```python
from bayesian_priors.data_fetching import (
    fetch_fred_series,           # For US and Mexico
    fetch_china_wages_manufacturing,  # For China via World Bank
)
```

### 2. Normalize to Baseline
All series (whether indices, hourly rates, or per capita values) are normalized to the baseline cost per lamp:
```python
normalized = (series / series.mean()) * baseline_per_lamp
```

This ensures all data maps to your cost structure (dollars per lamp).

### 3. Fit Bayesian Posterior
Normal-Inverse-Gamma (N-IG) posterior is fitted to the normalized data:
```python
from bayesian_priors.parameter_estimators import estimate_labor_posterior

posterior = estimate_labor_posterior("China", baseline=1.20)
```

### 4. Generate Samples
Student-t predictive distribution (heavy tails handle uncertainty):
```python
samples = posterior.sample_predictive(n_runs=10000)
```

---

## Why These Sources?

### ✅ Advantages

1. **100% FREE** - No payment, no API keys, no rate limits
2. **Reliable** - Official government/international organization data
3. **Historical** - Sufficient data for robust Bayesian inference
4. **Maintained** - Regularly updated by institutions
5. **No Authentication** - Simple HTTP requests

### 🔄 Data Quality

- **US/Mexico**: Direct labor market data (hourly earnings)
- **China**: Proxy data (GDP per capita) - highly correlated with wages
- All sources provide sufficient historical data for posterior estimation

---

## Fallback Strategy

If any data source fails (network issues, API changes):

1. System detects empty series
2. Logs warning message
3. Falls back to **weakly-informative prior** centered at baseline
4. Simulation continues without interruption

```python
# Automatic fallback in estimate_labor_posterior()
if series is None or len(series) < 2:
    print(f"⚠️  Fallback: using weak N-IG prior at ${baseline:.2f}")
    return NormalPosterior(
        mu=baseline,
        kappa=20.0,  # weak confidence
        alpha=10.0,
        beta=(baseline * 0.10) ** 2 * 10.0
    )
```

---

## Cost Comparison

### Paid Alternative: TradingEconomics
- **Cost**: $500-$5,000/month depending on tier
- **Benefit**: More granular indicators (e.g., "Wages in Manufacturing")
- **Verdict**: ❌ Not necessary - free alternatives work well

### Our Solution: FRED + World Bank
- **Cost**: $0
- **Benefit**: Sufficient data quality for Bayesian modeling
- **Verdict**: ✅ Excellent for this use case

---

## Testing

Run the integration tests to verify all data sources:

```bash
python -c "
from bayesian_priors.parameter_estimators import estimate_labor_posterior

for country in ['US', 'Mexico', 'China']:
    baseline = {'US': 3.50, 'Mexico': 2.20, 'China': 1.20}[country]
    post = estimate_labor_posterior(country, baseline)
    print(f'{country}: ${post.mu:.2f} ± ${post.sample_predictive(100).std():.2f}')
"
```

Expected output:
```
US: $3.50 ± $0.09
Mexico: $2.20 ± $0.11
China: $1.20 ± $0.32
```

---

## Need Different Data?

### To add a new country:

1. Find a free data source (FRED, World Bank, OECD, ILO)
2. Add fetcher in `bayesian_priors/data_fetching.py`
3. Add mapping in `estimate_labor_posterior()` in `parameter_estimators.py`
4. Test with sample baseline value

### World Bank has 1000+ indicators:
Browse at: https://data.worldbank.org/indicator

Common labor indicators:
- `SL.UEM.TOTL.NE.ZS` - Unemployment rate
- `SL.TLF.TOTL.IN` - Labor force, total
- `NY.GDP.PCAP.PP.KD` - GDP per capita, PPP (constant)

---

## References

- **FRED**: https://fred.stlouisfed.org/
- **World Bank Data**: https://data.worldbank.org/
- **World Bank API Docs**: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
- **ILO Statistics**: https://ilostat.ilo.org/
- **OECD Data**: https://data.oecd.org/

---

**Last Updated**: October 2025  
**Status**: ✅ All sources operational and free

