"""
API Status Tests - Check which external data sources are working.

This test suite specifically checks which APIs are accessible and returning data,
regardless of whether fallback mechanisms work. This helps identify which APIs
need attention or alternative solutions.

CRITICAL: These tests report API failures even if fallbacks work.
The goal is to know which APIs to fix, not just whether the code runs.
"""

import pandas as pd
import pytest
from bayesian_priors.data_fetching import (
    fetch_fred_series,
    fetch_worldbank_indicator,
    fetch_china_wages_manufacturing,
)


class TestFREDAPIStatus:
    """Test FRED API accessibility for all series used in the project."""
    
    def test_ppi_plastics_for_raw_materials(self):
        """Test PPI Plastics (raw materials proxy) - PCU325211325211P"""
        series = fetch_fred_series("PCU325211325211P", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: PPI Plastics (PCU325211325211P) not available")
        else:
            print(f"✅ FRED PPI Plastics: {len(series)} data points retrieved")
    
    def test_us_labor_manufacturing_wages(self):
        """Test US Manufacturing Average Hourly Earnings - CES3000000003"""
        series = fetch_fred_series("CES3000000003", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: US Manufacturing Wages (CES3000000003) not available")
        else:
            print(f"✅ FRED US Manufacturing Wages: {len(series)} data points retrieved")
    
    def test_mexico_labor_earnings_index(self):
        """Test Mexico Hourly Earnings Index - LCEAMN01MXM661S"""
        series = fetch_fred_series("LCEAMN01MXM661S", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Mexico Earnings Index (LCEAMN01MXM661S) not available")
        else:
            print(f"✅ FRED Mexico Earnings Index: {len(series)} data points retrieved")
    
    def test_logistics_truck_freight(self):
        """Test Truck Freight PPI - PCU4841214841212"""
        series = fetch_fred_series("PCU4841214841212", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Truck Freight PPI (PCU4841214841212) not available")
        else:
            print(f"✅ FRED Truck Freight PPI: {len(series)} data points retrieved")
    
    def test_logistics_ocean_freight(self):
        """Test Ocean Freight PPI - PCU483111483111"""
        series = fetch_fred_series("PCU483111483111", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Ocean Freight PPI (PCU483111483111) not available")
        else:
            print(f"✅ FRED Ocean Freight PPI: {len(series)} data points retrieved")
    
    def test_logistics_air_freight(self):
        """Test Air Freight Index - IC1312"""
        series = fetch_fred_series("IC1312", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Air Freight Index (IC1312) not available")
        else:
            print(f"✅ FRED Air Freight Index: {len(series)} data points retrieved")
    
    def test_fx_mexico_peso(self):
        """Test Mexico FX Rate (MXN/USD) - DEXMXUS"""
        series = fetch_fred_series("DEXMXUS", months=12)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Mexico FX Rate (DEXMXUS) not available")
        else:
            print(f"✅ FRED Mexico FX Rate: {len(series)} data points retrieved")
    
    def test_fx_china_yuan(self):
        """Test China FX Rate (CNY/USD) - DEXCHUS"""
        series = fetch_fred_series("DEXCHUS", months=12)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: China FX Rate (DEXCHUS) not available")
        else:
            print(f"✅ FRED China FX Rate: {len(series)} data points retrieved")
    
    def test_us_indirect_eci_office(self):
        """Test US Employment Cost Index Office/Admin - ECIOCC52"""
        series = fetch_fred_series("ECIOCC52", months=48)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: US ECI Office/Admin (ECIOCC52) not available")
        else:
            print(f"✅ FRED US ECI Office/Admin: {len(series)} data points retrieved")
    
    def test_us_electricity_price(self):
        """Test US Electricity Price ($/kWh) - APU000072610"""
        series = fetch_fred_series("APU000072610", months=48)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: US Electricity Price (APU000072610) not available")
        else:
            print(f"✅ FRED US Electricity Price: {len(series)} data points retrieved")
    
    def test_mexico_electricity_cpi_energy(self):
        """Test Mexico CPI Energy - CPGRLE01MXQ661N"""
        series = fetch_fred_series("CPGRLE01MXQ661N", months=48)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: Mexico CPI Energy (CPGRLE01MXQ661N) not available")
        else:
            print(f"✅ FRED Mexico CPI Energy: {len(series)} data points retrieved")
    
    def test_china_electricity_cpi_energy(self):
        """Test China CPI Energy - CPGRLE01CNQ661N"""
        series = fetch_fred_series("CPGRLE01CNQ661N", months=48)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: China CPI Energy (CPGRLE01CNQ661N) not available")
        else:
            print(f"✅ FRED China CPI Energy: {len(series)} data points retrieved")
    
    def test_us_depreciation_machinery_ppi(self):
        """Test US Machinery PPI - PCU333999333999"""
        series = fetch_fred_series("PCU333999333999", months=60)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: US Machinery PPI (PCU333999333999) not available")
        else:
            print(f"✅ FRED US Machinery PPI: {len(series)} data points retrieved")
    
    def test_us_working_capital_fed_funds(self):
        """Test US Fed Funds Rate - FEDFUNDS"""
        series = fetch_fred_series("FEDFUNDS", months=24)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: US Fed Funds Rate (FEDFUNDS) not available")
        else:
            print(f"✅ FRED US Fed Funds Rate: {len(series)} data points retrieved")
    
    def test_china_working_capital_interbank_rate(self):
        """Test China 3-Month Interbank Rate - IR3TIB01CNM156N"""
        series = fetch_fred_series("IR3TIB01CNM156N", months=60)
        
        if len(series) < 2:
            pytest.fail("❌ FRED API FAILURE: China Interbank Rate (IR3TIB01CNM156N) not available")
        else:
            print(f"✅ FRED China Interbank Rate: {len(series)} data points retrieved")


class TestWorldBankAPIStatus:
    """Test World Bank API accessibility for all indicators used in the project."""
    
    def test_china_gdp_per_capita_for_wages(self):
        """Test China GDP per capita (wage proxy) - NY.GDP.PCAP.PP.CD"""
        series = fetch_worldbank_indicator("CHN", "NY.GDP.PCAP.PP.CD", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: China GDP per capita (NY.GDP.PCAP.PP.CD) not available")
        else:
            print(f"✅ World Bank China GDP per capita: {len(series)} data points retrieved")
    
    def test_mexico_indirect_cpi(self):
        """Test Mexico CPI - FP.CPI.TOTL"""
        series = fetch_worldbank_indicator("MEX", "FP.CPI.TOTL", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: Mexico CPI (FP.CPI.TOTL) not available")
        else:
            print(f"✅ World Bank Mexico CPI: {len(series)} data points retrieved")
    
    def test_china_indirect_cpi(self):
        """Test China CPI - FP.CPI.TOTL"""
        series = fetch_worldbank_indicator("CHN", "FP.CPI.TOTL", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: China CPI (FP.CPI.TOTL) not available")
        else:
            print(f"✅ World Bank China CPI: {len(series)} data points retrieved")
    
    def test_mexico_depreciation_investment_price(self):
        """Test Mexico Investment Price Level - PL.ITM.PLI"""
        series = fetch_worldbank_indicator("MEX", "PL.ITM.PLI", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: Mexico Investment Price (PL.ITM.PLI) not available")
        else:
            print(f"✅ World Bank Mexico Investment Price: {len(series)} data points retrieved")
    
    def test_china_depreciation_investment_price(self):
        """Test China Investment Price Level - PL.ITM.PLI"""
        series = fetch_worldbank_indicator("CHN", "PL.ITM.PLI", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: China Investment Price (PL.ITM.PLI) not available")
        else:
            print(f"✅ World Bank China Investment Price: {len(series)} data points retrieved")
    
    def test_mexico_working_capital_lending_rate(self):
        """Test Mexico Lending Rate - FR.INR.LEND"""
        series = fetch_worldbank_indicator("MEX", "FR.INR.LEND", years=15)
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: Mexico Lending Rate (FR.INR.LEND) not available")
        else:
            print(f"✅ World Bank Mexico Lending Rate: {len(series)} data points retrieved")
    
    def test_china_labor_wages_function(self):
        """Test China wages fetch function (composite indicator)"""
        series = fetch_china_wages_manufacturing()
        
        if len(series) < 2:
            pytest.fail("❌ WORLD BANK API FAILURE: China wages (composite) not available")
        else:
            print(f"✅ World Bank China Wages (composite): {len(series)} data points retrieved")


class TestAPIStatusSummary:
    """Generate comprehensive API status summary."""
    
    def test_generate_api_status_report(self):
        """Generate a comprehensive report of all API statuses."""
        
        print("\n" + "=" * 80)
        print("API STATUS REPORT - BAYESIAN PRIORS DATA SOURCES")
        print("=" * 80)
        
        results = {
            "FRED APIs": {},
            "World Bank APIs": {}
        }
        
        # Test all FRED series
        fred_series = {
            "Raw Materials (PPI Plastics)": "PCU325211325211P",
            "US Labor (Mfg Wages)": "CES3000000003",
            "Mexico Labor (Earnings Index)": "LCEAMN01MXM661S",
            "Logistics (Truck Freight)": "PCU4841214841212",
            "Logistics (Ocean Freight)": "PCU483111483111",
            "Logistics (Air Freight)": "IC1312",
            "FX (Mexico Peso)": "DEXMXUS",
            "FX (China Yuan)": "DEXCHUS",
            "US Indirect (ECI Office)": "ECIOCC52",
            "US Electricity ($/kWh)": "APU000072610",
            "Mexico Electricity (CPI Energy)": "CPGRLE01MXQ661N",
            "China Electricity (CPI Energy)": "CPGRLE01CNQ661N",
            "US Depreciation (Machinery PPI)": "PCU333999333999",
            "US Working Capital (Fed Funds)": "FEDFUNDS",
            "China Working Capital (Interbank)": "IR3TIB01CNM156N",
        }
        
        print("\n📊 FRED API Status:")
        print("-" * 80)
        for name, series_id in fred_series.items():
            try:
                series = fetch_fred_series(series_id, months=24)
                if len(series) >= 2:
                    results["FRED APIs"][name] = "✅ OK"
                    print(f"  ✅ {name:50s} [{series_id:20s}] {len(series):3d} points")
                else:
                    results["FRED APIs"][name] = "❌ NO DATA"
                    print(f"  ❌ {name:50s} [{series_id:20s}] NO DATA")
            except Exception as e:
                results["FRED APIs"][name] = f"❌ ERROR"
                print(f"  ❌ {name:50s} [{series_id:20s}] ERROR: {str(e)[:30]}")
        
        # Test all World Bank indicators
        wb_indicators = {
            "China Labor (GDP per capita)": ("CHN", "NY.GDP.PCAP.PP.CD"),
            "Mexico Indirect (CPI)": ("MEX", "FP.CPI.TOTL"),
            "China Indirect (CPI)": ("CHN", "FP.CPI.TOTL"),
            "Mexico Depreciation (Investment Price)": ("MEX", "PL.ITM.PLI"),
            "China Depreciation (Investment Price)": ("CHN", "PL.ITM.PLI"),
            "Mexico Working Capital (Lending Rate)": ("MEX", "FR.INR.LEND"),
        }
        
        print("\n🌍 World Bank API Status:")
        print("-" * 80)
        for name, (country, indicator) in wb_indicators.items():
            try:
                series = fetch_worldbank_indicator(country, indicator, years=15)
                if len(series) >= 2:
                    results["World Bank APIs"][name] = "✅ OK"
                    print(f"  ✅ {name:50s} [{indicator:20s}] {len(series):3d} points")
                else:
                    results["World Bank APIs"][name] = "❌ NO DATA"
                    print(f"  ❌ {name:50s} [{indicator:20s}] NO DATA")
            except Exception as e:
                results["World Bank APIs"][name] = f"❌ ERROR"
                print(f"  ❌ {name:50s} [{indicator:20s}] ERROR: {str(e)[:30]}")
        
        # Summary statistics
        fred_ok = sum(1 for v in results["FRED APIs"].values() if "✅" in v)
        fred_total = len(results["FRED APIs"])
        wb_ok = sum(1 for v in results["World Bank APIs"].values() if "✅" in v)
        wb_total = len(results["World Bank APIs"])
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"FRED APIs:       {fred_ok}/{fred_total} working ({fred_ok/fred_total*100:.1f}%)")
        print(f"World Bank APIs: {wb_ok}/{wb_total} working ({wb_ok/wb_total*100:.1f}%)")
        print(f"TOTAL:           {fred_ok + wb_ok}/{fred_total + wb_total} working ({(fred_ok + wb_ok)/(fred_total + wb_total)*100:.1f}%)")
        print("=" * 80)
        
        # Collect failures
        failures = []
        for source, apis in results.items():
            for name, status in apis.items():
                if "❌" in status:
                    failures.append(f"{source}: {name}")
        
        if failures:
            print("\n⚠️  APIS REQUIRING ATTENTION:")
            print("-" * 80)
            for failure in failures:
                print(f"  • {failure}")
            print("=" * 80)
            
            # Don't fail the test, just report
            print("\n⚠️  Warning: Some APIs are not working. See report above.")
        else:
            print("\n✅ All APIs are working correctly!")
            print("=" * 80)


class TestDataQuality:
    """Test data quality for APIs that are working."""
    
    def test_data_has_no_all_zeros(self):
        """Test that data series are not all zeros (would indicate bad data)."""
        series = fetch_fred_series("FEDFUNDS", months=24)
        
        if len(series) >= 2:
            # Check that not all values are zero
            assert not (series == 0).all(), \
                "Fed Funds data is all zeros (data quality issue)"
    
    def test_data_has_reasonable_variance(self):
        """Test that data has reasonable variance (not all same value)."""
        series = fetch_fred_series("DEXMXUS", months=12)
        
        if len(series) >= 3:
            # Check that standard deviation is non-zero
            std = series.std()
            assert std > 0, \
                "FX data has zero variance (data quality issue)"
    
    def test_data_timestamps_are_chronological(self):
        """Test that data timestamps are in chronological order."""
        series = fetch_fred_series("CES3000000003", months=24)
        
        if len(series) >= 2:
            # Index should be sorted
            assert series.index.is_monotonic_increasing, \
                "Data timestamps are not chronological"


if __name__ == "__main__":
    # Run with verbose output to see all status messages
    pytest.main([__file__, "-v", "-s"])

