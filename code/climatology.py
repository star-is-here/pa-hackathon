import pandas as pd
import numpy as np
import ujson as json
from shapely.geometry import Point, shape
from tqdm import tqdm

tqdm.pandas(desc="Gridding Accidents")



def featPoint(x, y, id):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [x, y]},
        "properties": {"label": "%s,%s" % (x, y)}
        }
    return feat


def featPolygon(bbox, uid):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [bbox]},
        "properties": {
            "label": "(%s, %s), (%s, %s)" % (bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]),
            "uid": uid
            }
        }
    return feat


def grid(minx, miny, maxx, maxy, n_splits):
    geojson = {
      "type": "FeatureCollection",
    }

    features = []

    x_increment = (maxx - minx) / n_splits
    y_increment = abs(maxy - miny) / n_splits

    for i, x in enumerate(np.arange(minx, maxx, x_increment)):
        for j, y in enumerate(np.arange(miny, maxy, y_increment)):
            tLx, bLx = (x, x)
            tLy, tRy = (y, y)
            tRx, bRx = (x + x_increment, x + x_increment)
            bLy, bRy = (y + y_increment, y + y_increment)
            bbox = [[tLx, tLy], [tRx, tRy], [bRx, bRy], [bLx, bLy], [tLx, tLy]]
            features.append(featPolygon(bbox, '%02d%02d' % (i, j)))

    geojson["features"] = features

    # return json.dumps(geojson, indent=4)
    return geojson


def getID(lon, lat, lon_range, lat_range):
    if len([j for j, y in enumerate(lat_range) if y < lat]) == 0:
        print(lon, lat)
    xid = max([i for i, x in enumerate(lon_range) if x < lon])
    yid = max([j for j, y in enumerate(lat_range) if y < lat])
    return '%02d%02d' % (xid, yid)


def addProperties(geojson, df):
    for i, feat in enumerate(tqdm(geojson['features'])):
        try:
            feat['properties']['accidents'] = df.loc[feat['properties']['uid'], 'Crash Record Number']
            feat['properties']['fatality'] = df.loc[feat['properties']['uid'], 'Fatality Count']
            feat['properties']['injury'] = df.loc[feat['properties']['uid'], 'Injury Count']
            feat['properties']['prob_acc'] = feat['properties']['accidents'] * np.random.uniform(.7, .9, 1)[0]
            feat['properties']['prob_fatal'] = feat['properties']['fatality'] * np.random.uniform(.7, .9, 1)[0]
            feat['properties']['prob_injury'] = feat['properties']['injury'] * np.random.uniform(.7, .9, 1)[0]
            geojson['features'][i] = feat
        except KeyError:
            feat['properties']['accidents'] = 0
            feat['properties']['fatality'] = 0
            feat['properties']['injury'] = 0
            feat['properties']['prob_acc'] = 0
            feat['properties']['prob_fatal'] = 0
            feat['properties']['prob_injury'] = 0
    return geojson

with open('data/PA.geojson', 'r') as f:
    pa = json.load(f)
    paLon = [x[0] for x in pa['geometry']['coordinates'][0]]
    paLat = [x[1] for x in pa['geometry']['coordinates'][0]]
    maxLat, minLat = (max(paLat), min(paLat))
    maxLon, minLon = (max(paLon), min(paLon))

n_splits = 100
lon_increment = (maxLon - minLon) / n_splits
lat_increment = abs(maxLat - minLat) / n_splits
lon_range = np.arange(minLon, maxLon, lon_increment)
lat_range = np.arange(minLat, maxLat, lat_increment)
gridCrashes = grid(minLon, minLat, maxLon, maxLat, n_splits)

# crashes = pd.read_csv('data/Crash_Data__1997_to_Current__Transportation.csv')
# crashes.to_msgpack('data/Crash_Data__1997_to_Current__Transportation.msg')
# crashes = pd.read_csv('data/Crash_Data__1997_to_Current__Transportation_1000.csv')
crashes = pd.read_msgpack('data/Crash_Data__1997_to_Current__Transportation.msg')

filterNA = crashes['Longitude (Decimal)'].notnull() &\
    crashes['Latitude (Decimal)'].notnull() &\
    (crashes['Longitude (Decimal)'] > minLon) &\
    (crashes['Latitude (Decimal)'] > minLat) &\
    (crashes['Longitude (Decimal)'] < maxLon) &\
    (crashes['Latitude (Decimal)'] < maxLat)
crashes['uid'] = crashes[filterNA].progress_apply(
    lambda row: getID(row['Longitude (Decimal)'], row['Latitude (Decimal)'], lon_range, lat_range),
    axis=1
    )
crashesGroup = crashes.groupby('uid').agg({
  'Crash Record Number': 'count',
  'Fatality Count': 'mean',
  'Injury Count': 'mean'
})
maxCrashes = crashesGroup['Crash Record Number'].max()
maxFatal = crashesGroup['Fatality Count'].max()
maxInjury = crashesGroup['Injury Count'].max()
crashesGroup['Crash Record Number'] = crashesGroup['Crash Record Number'] / maxCrashes
crashesGroup['Fatality Count'] = crashesGroup['Fatality Count'] / maxFatal
crashesGroup['Injury Count'] = crashesGroup['Injury Count'] / maxInjury
gridCrashes = addProperties(gridCrashes, crashesGroup)

with open('data/grid.geojson', 'w') as f:
    json.dump(gridCrashes, f)
