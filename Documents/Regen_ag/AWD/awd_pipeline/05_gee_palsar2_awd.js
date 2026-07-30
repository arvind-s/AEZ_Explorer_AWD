/**
 * AWD detection using PALSAR-2 ScanSAR (JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR) in Google Earth Engine.
 *
 * Paste this into the GEE Code Editor: https://code.earthengine.google.com
 * (requires a free Earth Engine account -- this runs entirely on Google's
 * servers under your own login, so it is NOT affected by the network
 * restrictions that blocked the Planetary Computer / Sentinel-1 download
 * path in the sandbox this was built in.)
 *
 * Why PALSAR-2 for AWD instead of / in addition to Sentinel-1:
 * L-band (PALSAR-2) penetrates the rice canopy better than C-band
 * (Sentinel-1) and can sense sub-canopy water status even after canopy
 * closure -- this is the reason the AWD literature (Arai et al. 2022;
 * the ALOS-2 three-component + IoT paper) uses ALOS/PALSAR specifically for
 * irrigation-status detection, not just rice mapping. The tradeoff, per the
 * roadmap doc, is temporal resolution: ScanSAR's revisit over a given AOI is
 * typically sparser than Sentinel-1's ~6-12 days, which may limit how many
 * drying cycles you can actually resolve in a season. CHECK THIS FIRST for
 * your AOI/date range using the collection.size() print below before
 * investing in the rest of the analysis.
 *
 * Data reference: https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR-2_Level2_2_ScanSAR
 * Calibration: gamma-naught (dB) = 10*log10(DN^2) - 83.0  (JAXA's documented formula for this collection)
 * Bands: HH, HV (terrain-flattened gamma-naught DN), LIN (local incidence angle), MSK (data-quality bitmask)
 * MSK values: 1=valid, 2=layover, 3=shadow, 4=ocean water, 5=invalid -- we keep only MSK==1.
 * NOTE: not all ScanSAR scenes are dual-pol -- some are HH-only (no HV band present
 * at all, not just empty). The script below calibrates HH on every scene and HV only
 * on the subset that actually has it (see section 3) to avoid a "Band pattern 'HV' did
 * not match any bands" error.
 */

// ---------------------------------------------------------------------
// 0. AOI -- replace this with your own drawn/imported geometry, or (better,
//    for real MRV-grade work) an uploaded field/block boundary asset:
//    var geometry = ee.FeatureCollection('projects/<your-project>/assets/<your-asset>').geometry();
// ---------------------------------------------------------------------
var geometry = ee.Geometry.Rectangle([76.65, 29.95, 77.35, 30.50]); // approx. Ambala district, Haryana bbox -- NOT a surveyed boundary, replace when you have one

// ---------------------------------------------------------------------
// 1. STEP ONE: confirm data availability before building anything else.
//    (This mirrors the check you already had in your snippet -- keep doing
//    this first for any new AOI/date range.)
// ---------------------------------------------------------------------
var checkStart = '2020-06-01';
var checkEnd = '2020-07-02';
var checkCollection = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
  .filterBounds(geometry)
  .filterDate(checkStart, checkEnd);
print('STEP 1 -- images in', checkStart, 'to', checkEnd, ':', checkCollection.size());
print('STEP 1 -- image list:', checkCollection);

// ---------------------------------------------------------------------
// 2. Full-season query for the actual AWD analysis. AWD is a seasonal
//    drying/re-flooding PATTERN (see roadmap doc), so we need the whole
//    kharif season here, not a one-month window.
// ---------------------------------------------------------------------
var seasonStart = '2020-06-01';
var seasonEnd = '2020-11-30';
var seasonCollection = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
  .filterBounds(geometry)
  .filterDate(seasonStart, seasonEnd)
  .sort('system:time_start');
print('STEP 2 -- images in full season', seasonStart, 'to', seasonEnd, ':', seasonCollection.size());
print('If this is 0-2 images, PALSAR-2 ScanSAR revisit is too sparse for cycle '
  + 'detection at this AOI/period -- treat it as a single/few-date sanity check '
  + 'against Sentinel-1 instead of a standalone time series.');

// ---------------------------------------------------------------------
// 3. Calibrate DN -> gamma-naught (dB) and mask out invalid/layover/shadow/
//    ocean pixels using the MSK band. This is the SAR equivalent of the
//    speckle-filter + calibration step in the Sentinel-1 pipeline
//    (02_calibrate_and_filter.py) -- mandatory, not optional, per the
//    roadmap's finding that unfiltered/uncalibrated data measurably hurts
//    classification accuracy.
//
//    IMPORTANT: not every ScanSAR scene is dual-pol. Some acquisitions are
//    HH-only (bands: HH, LIN, MSK -- no HV), per the collection's
//    'Polarizations' metadata (['HH'] vs ['HH','HV']). Calling
//    image.select('HV') on an HH-only scene throws
//    "Band pattern 'HV' did not match any bands." So: calibrate HH on the
//    FULL collection (every scene has it), and calibrate HV only on the
//    subset of scenes that actually have it.
// ---------------------------------------------------------------------
function calibrateHH(image) {
  var validMask = image.select('MSK').eq(1); // keep only "valid data"
  var hhDn = image.select('HH').updateMask(image.select('HH').gt(0));
  var hhDb = hhDn.pow(2).log10().multiply(10).subtract(83.0).rename('HH_dB');
  return image.addBands(hhDb)
    .updateMask(validMask)
    .set('system:time_start', image.get('system:time_start'));
}

function calibrateHV(image) {
  var validMask = image.select('MSK').eq(1);
  var hvDn = image.select('HV').updateMask(image.select('HV').gt(0));
  var hvDb = hvDn.pow(2).log10().multiply(10).subtract(83.0).rename('HV_dB');
  return image.addBands(hvDb)
    .updateMask(validMask)
    .set('system:time_start', image.get('system:time_start'));
}

var calibrated = seasonCollection.map(calibrateHH); // HH_dB on every scene

var dualPolCollection = seasonCollection.filter(ee.Filter.listContains('Polarizations', 'HV'));
print('STEP 2b -- dual-pol (HH+HV) scenes in season:', dualPolCollection.size(),
  '(the rest are HH-only -- HH is still used for the main time series/model either way)');
var calibratedDualPol = dualPolCollection.map(calibrateHV); // HV_dB only where it exists

// ---------------------------------------------------------------------
// 4. Visualization (adapted from your original snippet, now on calibrated
//    dB values instead of raw DN so the min/max are physically meaningful).
//    Typical L-band gamma0 range for rice paddies: roughly -20 dB (flooded)
//    to -6 dB (dense vegetative canopy) -- adjust after inspecting your
//    AOI's actual histogram (Inspector tool in the Code Editor). Uses
//    `calibrated` (HH_dB), which every scene has, so this loop won't hit
//    the missing-band error regardless of polarization mix.
// ---------------------------------------------------------------------
Map.centerObject(geometry, 10);
Map.addLayer(calibrated.select('HH_dB'), {min: -20, max: -6, palette: ['blue', 'white', 'green']},
  'HH gamma0 dB (' + seasonStart + ' to ' + seasonEnd + ')');

var imgList = calibrated.toList(calibrated.size());
var imgCount = calibrated.size().getInfo();
for (var i = 0; i < imgCount; i++) {
  var img = ee.Image(imgList.get(i));
  var dateStr = img.date().format('YYYY-MM-dd').getInfo();
  Map.addLayer(img.select('HH_dB'), {min: -20, max: -6, palette: ['blue', 'white', 'green']},
    'HH_dB ' + dateStr, false); // hidden by default, toggle individually to inspect
}

// ---------------------------------------------------------------------
// 5. Zonal (AOI-average) time series extraction -- produces the same
//    schema as the Sentinel-1 pipeline's s1_aoi_timeseries.csv
//    (date, orbit_state, polarization, mean, median, p10, p90, n_valid),
//    so it's a drop-in input to 03_wetness_index_model.py without any code
//    changes on the Python side -- just point --polarization at HH or HV.
// ---------------------------------------------------------------------
function extractSeries(imgCol, bandName, polLabel) {
  return ee.FeatureCollection(imgCol.map(function(image) {
    var stats = image.select([bandName]).reduceRegion({
      reducer: ee.Reducer.mean()
        .combine({reducer2: ee.Reducer.median(), sharedInputs: true})
        .combine({reducer2: ee.Reducer.percentile([10, 90]), sharedInputs: true})
        .combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
      geometry: geometry,
      scale: 25,
      maxPixels: 1e13,
      bestEffort: true
    });
    return ee.Feature(null, {
      date: image.date().format('YYYY-MM-dd'),
      orbit_state: image.get('PassDirection'),
      polarization: polLabel,
      mean: stats.get(bandName + '_mean'),
      median: stats.get(bandName + '_median'),
      p10: stats.get(bandName + '_p10'),
      p90: stats.get(bandName + '_p90'),
      n_valid: stats.get(bandName + '_count')
    });
  }));
}

var hhSeries = extractSeries(calibrated, 'HH_dB', 'HH');           // every scene
var hvSeries = extractSeries(calibratedDualPol, 'HV_dB', 'HV');    // dual-pol scenes only
var combinedSeries = hhSeries.merge(hvSeries);

// ---------------------------------------------------------------------
// 6. Quick look at the time series right here in the Code Editor, before
//    waiting on an export -- if this chart looks wrong (empty, flat,
//    outside a plausible dB range), fix that before exporting anything.
// ---------------------------------------------------------------------
var chart = ui.Chart.feature.byFeature({
  features: hhSeries.filter(ee.Filter.notNull(['mean'])),
  xProperty: 'date',
  yProperties: ['mean']
}).setChartType('LineChart')
  .setOptions({title: 'PALSAR-2 HH gamma0 (dB) -- AOI mean over time', vAxis: {title: 'dB'}});
print(chart);

// ---------------------------------------------------------------------
// 7. Export -- download this CSV, then run the existing Python model on it
//    exactly as with the Sentinel-1 pipeline:
//      python 03_wetness_index_model.py --timeseries palsar2_ambala_timeseries.csv --polarization HH
// ---------------------------------------------------------------------
Export.table.toDrive({
  collection: combinedSeries,
  description: 'palsar2_awd_timeseries_ambala',
  folder: 'AWD_exports',
  fileNamePrefix: 'palsar2_ambala_timeseries',
  fileFormat: 'CSV'
});
