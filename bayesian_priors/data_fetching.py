"""
Data Fetching Module

Retrieves real-world economic data from:
 - FRED (Federal Reserve Economic Data) via graph CSV (no key needed)
 - World Bank Open Data API (100% free, no key required)

These feed our Bayesian models (Normal–Inverse-Gamma → Student-t).
"""

import pandas as pd


def fetch_fred_series(series_id: str, months: int = 24) -> pd.Series:
    """Fetch economic time series from FRED API."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        # FRED uses 'observation_date' as the date column name
        df = pd.read_csv(url)

        # Check if we got valid data
        if df.empty or len(df.columns) < 2:
            print(f"⚠️  No data returned for {series_id}")
            return pd.Series(dtype=float)

        # Parse date column (usually 'observation_date')
        date_col = df.columns[0]
        value_col = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col)

        # Convert values to numeric and drop missing
        values = pd.to_numeric(df[value_col], errors="coerce").dropna()

        if len(values) == 0:
            print(f"⚠️  No valid numeric data for {series_id}")
            return pd.Series(dtype=float)

        # Keep only recent months
        cutoff = values.index.max() - pd.DateOffset(months=months)
        values = values[values.index > cutoff]

        return values

    except Exception as e:
        print(f"⚠️  Could not fetch {series_id}: {e}")
        return pd.Series(dtype=float)


# -----------------------------
# World Bank API (FREE, no key required)
# -----------------------------
def fetch_worldbank_indicator(country_code: str, indicator: str, years: int = 20) -> pd.Series:
    """
    Fetch a World Bank indicator as a pandas Series.
    World Bank API is completely FREE and requires no API key.
    
    Args:
        country_code: ISO country code (e.g., 'CHN' for China, 'USA' for US)
        indicator: World Bank indicator code
        years: Number of years of data to fetch
    
    Example indicators:
        - SL.UEM.TOTL.NE.ZS: Unemployment rate
        - NY.GDP.MKTP.CD: GDP (current US$)
        - Various labor indicators available
    """
    import json
    from urllib.request import urlopen
    
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/"
        f"indicator/{indicator}?format=json&per_page={years}"
    )
    
    try:
        with urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
        
        # World Bank returns [metadata, data_array]
        if len(data) < 2 or not data[1]:
            print(f"⚠️  No World Bank data for {indicator}")
            return pd.Series(dtype=float)
        
        records = data[1]
        
        # Extract year and value
        dates = []
        values = []
        for record in records:
            if record.get('value') is not None:
                dates.append(pd.Timestamp(f"{record['date']}-01-01"))
                values.append(float(record['value']))
        
        if not dates:
            return pd.Series(dtype=float)
        
        series = pd.Series(values, index=dates)
        series = series.sort_index()
        return series
        
    except Exception as e:
        print(f"⚠️  World Bank fetch failed for {country_code}/{indicator}: {e}")
        return pd.Series(dtype=float)


def fetch_china_wages_manufacturing() -> pd.Series:
    """
    Fetch China manufacturing wage/labor cost data using FREE World Bank API.
    
    Uses multiple indicators to construct a labor cost proxy:
    1. Try labor cost indicator
    2. Fallback to GDP per capita (correlates with wages)
    3. Fallback to unemployment (inverse proxy for labor market tightness)
    
    Returns normalized series suitable for posterior estimation.
    """
    # Try various World Bank indicators for China labor market
    # These are all FREE and require no API key
    
    # Option 1: GDP per capita (PPP) - correlates strongly with wages
    series = fetch_worldbank_indicator("CHN", "NY.GDP.PCAP.PP.CD", years=15)
    if len(series) >= 5:
        print(f"✓ Using China GDP per capita as wage proxy ({len(series)} years)")
        return series
    
    # Option 2: GNI per capita (Atlas method)
    series = fetch_worldbank_indicator("CHN", "NY.GNP.PCAP.CD", years=15)
    if len(series) >= 5:
        print(f"✓ Using China GNI per capita as wage proxy ({len(series)} years)")
        return series
    
    # If all fail, return empty (fallback to prior in estimator)
    print("⚠️  Could not fetch China labor data from World Bank")
    return pd.Series(dtype=float)


