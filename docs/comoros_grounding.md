# Comoros: institutions, decisions, and what the results say (material for Andrew's paragraphs)

Compiled 2026-09-05 from public sources; every statement carries its source. Nothing here is in the manuscript yet.

## 1. Who forecasts, who warns, who acts

- **ANACM / DTM.** The National Agency for Civil Aviation and Meteorology (ANACM, created 2017 by decree 17-024/PR,
  reorganised by decree 19-110/PR) hosts the Technical Directorate of Meteorology (DTM), the country's NMHS. DTM has
  no 24/7 alert service; a monitoring protocol starts when an extreme event is forecast within 72 h, produces an
  information note for the DGSC, and issues an alert if the event is confirmed.
  Source: SOFF/WMO Country Hydromet Diagnostics, Union of Comoros, July 2024, Element 6.1
  (https://www.un-soff.org/wp-content/uploads/2024/10/Rapport_CHD_Comores.pdf).
- **DGSC.** The Direction Générale de la Sécurité Civile is "the sole recipient of the weather alert and the only one
  authorized to communicate" to the public (radio and television); it triggers the alert phases through the Minister
  in charge of civil protection, COSEP (rescue operations and civil protection centre) and the regional CROSEP.
  Marine warnings cover strong swell, exceptional high tides, navigation and tsunami; storm surge is not a product.
  Source: same report, Element 6.1 and Figure 13; the 2010 national alert procedures are cited for the yellow
  cyclone alert of 23 April 2019 (OCHA Flash Update no. 1, Tropical Cyclone Kenneth, 24 April 2019,
  https://reliefweb.int/report/mozambique/southern-africa-tropical-cyclone-kenneth-flash-update-no-1-24-april-2019).
- **Regional forecast supply.** DTM builds its forecasts from web products of RSMC La Réunion (Météo-France),
  RSMC Pretoria, RIMES Bangkok and ECMWF; its PUMA satellite and model reception system has been non-operational for
  years, and it has "neither the capacity nor the resources to launch a local atmospheric model" (Element 5,
  maturity level 1). Source: same report, Elements 4 and 5.
- **Comorian Red Crescent / IFRC.** For Kenneth (April 2019) the IFRC released CHF 300,000 from the DREF on 29 April;
  the National Society mobilised about 100 volunteers for early-warning messaging and evacuation support in
  coordination with the DGSC and OCHA. Sources: IFRC Emergency Plan of Action MDRKM007 and Operation Update no. 1
  (https://reliefweb.int/report/comoros/comoros-tropical-cyclone-kenneth-emergency-plan-action-dref-n-mdrkm007-pkm010;
  https://reliefweb.int/report/comoros/comoros-tropical-cyclone-kenneth-emergency-plan-action-operation-update-n-1-emergency).
  A DREF operation also followed Cyclone Belna (December 2019, MDRKM008).
- **Strategy and anticipatory action.** National DRR strategy adopted 2014, updated 2022; a National DRR Strategy and
  Action Plan 2024-2030 is in force; in September 2025 the DGSC and the Comorian Red Crescent ran a multi-hazard
  tabletop simulation that tested early-warning responsibilities and anticipatory-action capacities.
  Source: UNDRR news, "Comoros charts a safer future with new disaster risk reduction strategy"
  (https://www.undrr.org/news/comoros-charts-safer-future-new-disaster-risk-reduction-strategy); SOFF report, Element 6.2.

## 2. Observations: there is no working gauge

- DTM "previously had a buoy and tide gauges for monitoring ocean conditions, but these instruments were either lost
  to the ocean or destroyed by cyclonic forces"; the country "lacks upper-air observations, marine observations, and
  remote sensing" (Element 3, maturity level 2). Installing tide gauges, buoys and marine radar is the report's first
  infrastructure recommendation. Source: SOFF report, Element 3 summary.
- The 2011 IHO capacity assessment already recorded no tide tables and no tidal observations for the Comoros
  (https://iho.int/uploads/user/Capacity%20Building/Reports%20Assessments/2011/4-Comoros-report.pdf).
- The Comoros are absent from GESLA-3. In our corpus the nearest gauges to Moroni are Zanzibar (764 km, training),
  Mombasa (938 km, validation), Lamu (1,081 km, test), Pointe La Rue (1,555 km, validation) and Réunion (1,639 km,
  training); no gauge sits inside the Mozambique Channel (`scripts/audit2_numbers.py` neighbourhood listing,
  2026-09-05).

## 3. Exposure and the 2019 event

- Population 742,287 (69% rural); observed sea-level rise 4 mm per year (SOFF report, Chapter 1, citing the Third
  National Communication). The SWIO Risk Assessment and Financing Initiative risk profile maps cyclone and flood
  exposure for the three islands (SOFF report, Element 6.3; GFDRR profile
  https://www.gfdrr.org/sites/default/files/publication/dra_comores_fr.pdf).
- Cyclone Kenneth, 24-25 April 2019: 7 dead, about 200 injured, about 20,000 displaced, 3,818 houses destroyed and
  7,013 damaged, one hospital flooded (UNICEF Comoros humanitarian situation reports no. 1 and no. 6,
  https://reliefweb.int/report/comoros/comoros-humanitarian-situation-report-no-1-cyclone-kenneth;
  https://reliefweb.int/report/comoros/comoros-humanitarian-situation-report-no-6-cyclone-kenneth-3-june-2019).
  Kenneth passed near Grande Comore with sustained winds forecast up to 190 km/h; storm surge was flagged for the
  Comoros and northern Mozambique on 25-26 April (OCHA Flash Update no. 1).

## 4. What the manuscript's results imply for this case (facts we can defend)

- The gauge-free configuration (EOT20 tides, reanalysis-forcing covariates, no water level) can run at Moroni today
  from open data; on our 84 test gauges it reaches NNSE 0.59 at 8 h and shows no detectable difference from the global
  hydrodynamic reanalysis on everyday accuracy, while the physics has the lower storm-window error. So without a gauge
  the model is a free second opinion, not a storm-surge warning system.
- One working gauge is the largest single term in the information ladder (+0.16 NNSE at 8 h, tier A to tier B), and
  the Mozambique Channel is an unrepresented regime in the training network, so a Moroni gauge would count twice:
  as the local anchor and as regime coverage for every coast that shares it. This is the same investment the SOFF
  diagnostic lists first.
- The decision that a surge forecast would inform is the DGSC's alert phase and evacuation order, reached through the
  DTM information note; the operational recipient of any guidance is the DTM forecast service, and the last-mile
  actor is the Comorian Red Crescent volunteer network that already carries warnings to households.
- Honest limits for this case: the model has been evaluated only under reanalysis forcing (season-scale real-forecast
  check under way, see `outputs/eval_gefs_season.log`), at gauges with long records, and the Comoros have none; a
  prospective test would need a gauge installed first (Discussion, next steps).
