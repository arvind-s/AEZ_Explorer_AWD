/**
 * Full AWD methodology, entirely in Google Earth Engine -- pixel-level
 * classification + acreage stats + GeoTIFF export. No download/Python
 * round-trip needed; the only outputs you pull out of GEE are the final
 * AWD/non-AWD raster (GeoTIFF) and an acreage stats table.
 *
 * This ports the same methodology as utils.py / 03_wetness_index_model.py
 * (Lovell 2019-style wetness index, threshold-based drying-cycle detection)
 * from "one time series for the whole AOI" to "one time series per pixel",
 * which is what lets us report actual AWD acreage instead of a single
 * AOI-wide score.
 *
 * Data: JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR (see 05_gee_palsar2_awd.js for
 * background on why L-band, and the HH-only-vs-dual-pol caveat). This script
 * uses HH only (guaranteed on every scene, per the error we hit in 05),
 * which keeps the pixel-level logic uniform across the whole collection.
 *
 * IMPORTANT CAVEATS (read before trusting the output):
 * 1. Not run against live GEE data yet -- there is no way to execute/debug
 *    Earth Engine code from the environment this was written in. Expect to
 *    paste this in, hit an error, and paste it back for a fix, same as we
 *    just did with 05_gee_palsar2_awd.js's HV band issue.
 * 2. The per-pixel "drying cycle" count below is a SIMPLIFICATION of the
 *    sequential wet->dry->rewet state machine in utils.detect_drying_cycles
 *    (Python). Earth Engine doesn't have a native per-pixel sequential loop,
 *    so this counts "dry-at-t immediately followed by wet-at-t+1" events
 *    using shifted-array comparison instead -- close in spirit (same idea
 *    Lovell's own change-detection approach uses: compare consecutive time
 *    steps), but not a byte-for-byte port. Validate against known AWD/
 *    non-AWD fields before trusting absolute acreage numbers.
 * 3. Speckle filtering here is a focal-median smoothing, simpler than the
 *    adaptive Refined Lee filter in utils.py -- fine for a first pass, but
 *    a known simplification.
 * 4. No rice-specific land cover mask is applied -- "ever flooded" (based on
 *    the wetness index) is used as a rough paddy/non-paddy proxy. This will
 *    misclassify other seasonally-flooded land (e.g. wetlands, fish ponds)
 *    as candidate paddy. Adding a real rice mask (WorldCereal/GloRice, per
 *    the roadmap doc) is the natural next improvement.
 * 5. Ambala AOI below is still an approximate bounding box, not a surveyed
 *    boundary -- swap in a real asset when you have one.
 * 6. Start with a SMALL AOI (a village/block, not a whole district) for your
 *    first successful run -- per-pixel array operations over a large area
 *    are much more expensive than the AOI-averaged version in
 *    05_gee_palsar2_awd.js, and it's easier to debug logic errors on a small
 *    area before scaling up.
 */

// ---------------------------------------------------------------------
// 0. AOI + season + model parameters
// ---------------------------------------------------------------------
var geometry = ee.Geometry.Rectangle([76.65, 29.95, 77.35, 30.50]); // approx. Ambala bbox -- replace with a real boundary, ideally something small for the first run
var seasonStart = '2020-06-01';
var seasonEnd = '2020-11-30';

var WET_THRESHOLD = 0.3;      // WI >= this counts as "flooded" (same default as utils.py)
var DRY_THRESHOLD = -0.1;     // WI <= this counts as "dried down"
var REFERENCE_CYCLES = 5;     // cycles/season considered "fully AWD" -> likelihood 1.0
var AWD_LIKELIHOOD_THRESHOLD = 0.4; // >= this -> classified AWD
var EXPORT_SCALE = 25;        // meters; native PALSAR-2 ScanSAR resolution

// ---------------------------------------------------------------------
// 1. Load, filter, calibrate (HH only -- see caveats above), mask invalid
//    pixels, despeckle.
// ---------------------------------------------------------------------
var raw = ee.ImageCollection('JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR')
  .filterBounds(geometry)
  .filterDate(seasonStart, seasonEnd)
  .sort('system:time_start');
print('Images in season:', raw.size());
print('If this is fewer than ~5-6, per-pixel drying-cycle counts below will '
  + 'be unreliable -- ScanSAR revisit may be too sparse for this AOI/period.');

function calibrateHH(image) {
  var validMask = image.select('MSK').eq(1);
  var hhDn = image.select('HH').updateMask(image.select('HH').gt(0));
  var hhDb = hhDn.pow(2).log10().multiply(10).subtract(83.0).rename('HH_dB');
  return hhDb.updateMask(validMask).copyProperties(image, ['system:time_start']);
}

function despeckle(image) {
  // Simplified focal-median smoothing (see caveat 3 above).
  return image.focal_median({radius: 1.5, kernelType: 'square', units: 'pixels'})
    .rename(image.bandNames())
    .copyProperties(image, ['system:time_start']);
}

var hhCollection = raw.map(calibrateHH).map(despeckle);

// ---------------------------------------------------------------------
// 2. Per-pixel Wetness Index (Lovell 2019 style): rescale each pixel's own
//    seasonal min/max to -1..+1, using ImageCollection.min()/.max() as the
//    per-pixel reducers across the whole season -- this is the direct GEE
//    equivalent of utils.wetness_index()'s per-series min/max rescaling.
// ---------------------------------------------------------------------
var minImg = hhCollection.min();
var maxImg = hhCollection.max();
var range = maxImg.subtract(minImg);
var validRange = range.gt(0); // guard divide-by-zero for perfectly flat pixels

var wiCollection = hhCollection.map(function(img) {
  var wi = ee.Image(1).subtract(
    ee.Image(2).multiply(img.subtract(minImg)).divide(range)
  ).updateMask(validRange).rename('WI');
  return wi.copyProperties(img, ['system:time_start']);
});

// ---------------------------------------------------------------------
// 3. Per-pixel drying-cycle count via shifted-array comparison (see caveat 2).
//    "Re-flood event" = pixel was <= DRY_THRESHOLD at time t and >= WET_THRESHOLD
//    at time t+1. Sum of re-flood events across the season ~= number of
//    drying/re-flood cycles for that pixel.
// ---------------------------------------------------------------------
// ImageCollection.toArray() on a single-band collection produces a 2-D
// array per pixel: axis 0 = image/time sequence, axis 1 = band (size 1,
// since wiCollection only has the 'WI' band). arrayProject([0]) collapses
// that redundant band axis so we get a plain 1-D (time-only) array per
// pixel -- without this, arraySlice/arrayReduce/arrayGet below either
// operate on the wrong axis or throw "has 2 dimensions, but 'position' has
// 1 bands" from arrayGet expecting one index per remaining dimension.
var wiArray = wiCollection.toArray().arrayProject([0]);

var wetBool = wiArray.gte(WET_THRESHOLD);
var dryBool = wiArray.lte(DRY_THRESHOLD);

var dryAtT = dryBool.arraySlice(0, 0, -1);   // t = 0 .. n-2
var wetAtTplus1 = wetBool.arraySlice(0, 1);  // t = 1 .. n-1, aligned to dryAtT

var refloodEvents = dryAtT.multiply(wetAtTplus1); // 1 where a re-flood happened between t and t+1
var numCycles = refloodEvents.arrayReduce({reducer: ee.Reducer.sum(), axes: [0]}).arrayGet([0]).rename('n_cycles');

var awdLikelihood = numCycles.divide(REFERENCE_CYCLES).min(1).max(0).rename('awd_likelihood');

// "Ever flooded" proxy for "this pixel is plausibly a paddy at all" (see caveat 4).
var everFlooded = wiCollection.max().gte(WET_THRESHOLD).rename('ever_flooded');

// ---------------------------------------------------------------------
// 4. Classification
// ---------------------------------------------------------------------
var isAWD = awdLikelihood.gte(AWD_LIKELIHOOD_THRESHOLD).and(everFlooded).rename('is_awd');
var isContinuousFlood = everFlooded.and(isAWD.not()).rename('is_continuous_flood');

Map.centerObject(geometry, 11);
Map.addLayer(awdLikelihood.updateMask(everFlooded), {min: 0, max: 1, palette: ['white', 'orange', 'darkred']}, 'AWD likelihood');
Map.addLayer(isAWD.selfMask(), {palette: ['00ff00']}, 'Classified AWD');
Map.addLayer(isContinuousFlood.selfMask(), {palette: ['0000ff']}, 'Classified continuous flood');

// ---------------------------------------------------------------------
// 5. Acreage stats
// ---------------------------------------------------------------------
var pixelAreaHa = ee.Image.pixelArea().divide(10000);
var awdAreaImg = pixelAreaHa.updateMask(isAWD);
var floodAreaImg = pixelAreaHa.updateMask(isContinuousFlood);

var awdAreaStats = awdAreaImg.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: geometry, scale: EXPORT_SCALE, maxPixels: 1e13, bestEffort: true
});
var floodAreaStats = floodAreaImg.reduceRegion({
  reducer: ee.Reducer.sum(), geometry: geometry, scale: EXPORT_SCALE, maxPixels: 1e13, bestEffort: true
});

print('AWD acreage (ha):', awdAreaStats.get('area'));
print('Continuously-flooded / non-AWD acreage (ha):', floodAreaStats.get('area'));

var statsFeature = ee.FeatureCollection([
  ee.Feature(null, {
    'awd_area_ha': awdAreaStats.get('area'),
    'continuous_flood_area_ha': floodAreaStats.get('area'),
    'season_start': seasonStart,
    'season_end': seasonEnd,
    'wet_threshold': WET_THRESHOLD,
    'dry_threshold': DRY_THRESHOLD,
    'awd_likelihood_threshold': AWD_LIKELIHOOD_THRESHOLD,
    'n_images': raw.size()
  })
]);

// ---------------------------------------------------------------------
// 6. Exports -- the two deliverables you asked for. Both show up under the
//    "Tasks" tab in the Code Editor -- click Run on each to actually kick
//    them off (they don't run automatically just from executing the script).
// ---------------------------------------------------------------------
Export.image.toDrive({
  image: awdLikelihood.addBands(isAWD).addBands(isContinuousFlood).addBands(numCycles).clip(geometry),
  description: 'awd_classification_ambala',
  folder: 'AWD_exports',
  fileNamePrefix: 'awd_classification_ambala',
  region: geometry,
  scale: EXPORT_SCALE,
  maxPixels: 1e13
});

Export.table.toDrive({
  collection: statsFeature,
  description: 'awd_acreage_stats_ambala',
  folder: 'AWD_exports',
  fileNamePrefix: 'awd_acreage_stats_ambala',
  fileFormat: 'CSV'
});
