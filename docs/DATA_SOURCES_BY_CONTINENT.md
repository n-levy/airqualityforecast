# Global 100-City Dataset - Data Sources by Continent

## Overview
This document provides a comprehensive breakdown of data sources for the Global 100-City Air Quality Dataset, organized by continent with detailed access methods and standards.

---

## 🌍 EUROPE (20 Cities)
**Countries**: North Macedonia, Bosnia & Herzegovina, Bulgaria, Romania, Serbia, Poland, Czech Republic, Hungary, Italy, Greece, Spain, France, UK, Germany, Netherlands

### Data Sources
| Source Type | Provider | Method | API Key Required |
|-------------|----------|---------|------------------|
| **Ground Truth** | European Environment Agency (EEA) | Direct CSV downloads from Air Quality e-Reporting database | ❌ No |
| **Benchmark 1** | CAMS (Copernicus Atmosphere Monitoring Service) | Public API access via atmosphere.copernicus.eu | ❌ No |
| **Benchmark 2** | National monitoring networks | Government portals by country | ❌ No |

### Standards & Features
- **AQI Standard**: European Air Quality Index (EAQI) - Scale 1-6
- **Pollutants**: PM2.5, PM10, NO2, O3, SO2
- **Regional Features**: ETS indicators, cross-border transport, heating seasons, traffic restrictions

---

## 🇺🇸 NORTH AMERICA (20 Cities)
**Countries**: Mexico, USA, Canada

### Data Sources
| Source Type | Provider | Method | API Key Required |
|-------------|----------|---------|------------------|
| **Ground Truth** | EPA AirNow + Environment Canada | Public RSS feeds, open government data | ❌ No |
| **Benchmark 1** | NOAA air quality forecasts | Public scraping from airquality.weather.gov | ❌ No |
| **Benchmark 2** | State/provincial monitoring networks | Government portals (various) | ❌ No |

### Standards & Features
- **AQI Standards**:
  - US EPA AQI (0-500 scale) - USA cities
  - Canadian Air Quality Health Index (1-10+ scale) - Canadian cities
  - Mexican IMECA - Mexican cities
- **Pollutants**: PM2.5, PM10, NO2, O3, SO2
- **Regional Features**: Wildfire indicators, industrial emissions, interstate transport, seasonal inversions

---

## 🌏 ASIA (20 Cities)
**Countries**: India, Pakistan, China, Bangladesh, Thailand, Indonesia, Philippines, Vietnam, South Korea, Taiwan, Mongolia, Kazakhstan, Uzbekistan, Iran, Afghanistan

### Data Sources
| Source Type | Provider | Method | API Key Required |
|-------------|----------|---------|------------------|
| **Ground Truth** | National environmental agencies | Government portals (CPCB India, China MEE, etc.) | ❌ No |
| **Benchmark 1** | WAQI aggregated data | Public scraping from waqi.info (with attribution) | ❌ No |
| **Benchmark 2** | NASA satellite estimates | Public satellite APIs (earthdata.nasa.gov) | ❌ No |

### Standards & Features
- **AQI Standards**:
  - Indian National AQI (0-500 scale) - Indian cities
  - Chinese AQI (0-500 scale) - Chinese cities
  - Thai AQI - Thailand
  - Indonesian ISPU - Indonesia
  - Pakistani AQI - Pakistani cities
  - EPA AQI - Other Asian cities
- **Pollutants**: PM2.5, PM10, NO2, O3, SO2
- **Regional Features**: Monsoon indicators, dust storms, industrial activity, agricultural burning

---

## 🌍 AFRICA (20 Cities)
**Countries**: Chad, Egypt, Nigeria, Ghana, Sudan, Uganda, Kenya, Côte d'Ivoire, Mali, Burkina Faso, Senegal, DR Congo, Morocco, South Africa, Ethiopia, Tanzania, Algeria, Tunisia, Mozambique

### Data Sources
| Source Type | Provider | Method | API Key Required |
|-------------|----------|---------|------------------|
| **Ground Truth** | WHO Global Health Observatory | Open data portal (www.who.int/data/gho) | ❌ No |
| **Benchmark 1** | NASA MODIS satellite data | Public satellite APIs (modis.gsfc.nasa.gov) | ❌ No |
| **Benchmark 2** | Research networks (INDAAF, AERONET) | Research data repositories | ❌ No |

### Standards & Features
- **AQI Standard**: WHO Air Quality Guidelines
- **Pollutants**: PM2.5, PM10, NO2, O3 (limited SO2)
- **Regional Features**: Saharan dust, Harmattan effects, seasonal burning, mining activity

---

## 🌎 SOUTH AMERICA (20 Cities)
**Countries**: Peru, Chile, Brazil, Colombia, Bolivia, Argentina, Ecuador, Venezuela, Uruguay, Paraguay

### Data Sources
| Source Type | Provider | Method | API Key Required |
|-------------|----------|---------|------------------|
| **Ground Truth** | National environmental agencies | Government portals by country | ❌ No |
| **Benchmark 1** | NASA satellite estimates | Public satellite APIs (earthdata.nasa.gov) | ❌ No |
| **Benchmark 2** | Regional research networks | Research APIs and data repositories | ❌ No |

### Standards & Features
- **AQI Standards**:
  - EPA AQI adaptations - Most cities
  - Chilean ICA - Chilean cities
- **Pollutants**: PM2.5, PM10, NO2, O3, SO2
- **Regional Features**: Biomass burning, ENSO effects, altitude effects, Amazon influence

---

## 📊 Summary Statistics

| Continent | Cities | Countries | Unique AQI Standards | Primary Data Challenge |
|-----------|--------|-----------|---------------------|----------------------|
| Europe | 20 | 15 | 1 (EAQI) | High data availability |
| North America | 20 | 3 | 3 (EPA, Canadian, Mexican) | Good government data |
| Asia | 20 | 15 | 5+ (National standards) | Variable data quality |
| Africa | 20 | 19 | 1 (WHO) | Limited ground monitoring |
| South America | 20 | 10 | 2 (EPA adaptations, Chilean) | Moderate availability |
| **TOTAL** | **100** | **62** | **11+** | **Fully public access** |

---

## 🔑 Key Advantages

✅ **No Personal API Keys**: All data sources use public access methods
✅ **Standardized Benchmarks**: Consistent benchmark sources per continent
✅ **Local Standards**: Each city uses appropriate regional AQI calculation
✅ **Multi-Source Validation**: 2+ benchmarks per city for quality assurance
✅ **Scalable Architecture**: Easy to add more cities within each continent
✅ **Research Compliant**: Suitable for academic and commercial use

---

## 🚀 Implementation Status

**Current Status**: ✅ **FULLY SPECIFIED - READY FOR IMPLEMENTATION**

**Next Steps**:
1. Phase 1: Data Source Setup (1-2 weeks)
2. Phase 2: Data Collection Implementation (3-4 weeks)
3. Phase 3: Data Validation and QA (2-3 weeks)
4. Phase 4: Dataset Finalization (2-3 weeks)

---

*Last Updated: 2025-09-10*
*Version: 1.0*
*Total Implementation Time: 8-12 weeks*
