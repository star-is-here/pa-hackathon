import pandas as pd
import random
from pykrige.rk import RegressionKriging
from sklearn import linear_model, naive_bayes, gaussian_process, model_selection, metrics

# Full Dataset
# crash = pd.read_msgpack('Crash_Data__1997_to_Current__Transportation.msg')

# Split to Fatal/Non-Fatal
# fatalFilter = crash['Fatality Count'] > 0
# crash[fatalFilter].to_csv('fatalCrashes.csv', index=False)
# crash[~fatalFilter].to_csv('nonFatalCrashes.csv', index=False)
# crash[fatalFilter].to_msgpack('fatalCrashes.msg')
# crash[~fatalFilter].to_msgpack('nonFatalCrashes.msg')

# Load Fatal/Non-Fatal
n_nonfatal = 2600409
n_fatal = 25852
skip = sorted(random.sample(xrange(n_nonfatal), (n_nonfatal - n_fatal)))
fatal = pd.read_csv('fatalCrashes.csv')
nonfatal = pd.read_csv('nonFatalCrashes.csv', skiprows=skip)
nonfatal.columns = list(fatal)
crash = pd.concat([fatal, nonfatal], axis=0)

# First 1000
# crash = pd.read_csv('Crash_Data__1997_to_Current__Transportation_5000.csv')
# Select Subset of Variables

######################################################################################################################################################
# Clean Data
######################################################################################################################################################

keepVar = [
    'Crash Record Number',
    'County Code',
    'Municipality Code',
    'District Code',
    'Police Agency Code',
    'Crash Month',
    'Day of Week',
    'Time of Day',
    'Illumination',
    'Weather',
    'Road Condition',
    'Collision Type',
    'Work Zone Type',
    'Work Zone Location',
    'Intersection Type',
    'Traffic Control Device',
    'Location Type',
    'Urban / Rural',
    'Fatality Count',
    # 'Injury Count',
    'Person Count',
    'Total Unit Count',
    'School Bus Indicator',
    'School Zone Indicator',
    'Dispatch Time',
    'Construction Zone Speed Limit',
    'Estimated Hours Closed',
    'Lane Closed',
    'Lane Closure Direction',
    'Notify Highway Maintenance',
    'Special Jurisdiction',
    'Traffic Control Device Function',
    'Traffic Detour',
    'Workers Present',
    'Work Zone Close/Detour',
    'Work Zone Flagger',
    'Work Zone Law Officer',
    'Work Zone Closure',
    'Work Zone Moving',
    'Work Zone Other',
    'Work Zone Shoulder/Median',
    'Vehicle Count',
    'Automobile Count',
    'Motorcycle Count',
    'Bus Count',
    'Small Truck Count',
    'Heavy Truck Count',
    'SUV Count',
    'Van Count',
    'Bicycle Count',
    'Suspected Serious Injury Count',
    'Suspected Minor Injury Count',
    'Possible Injury Count',
    'Total Injury Count',
    'Unknown Injury Degree Count',
    'Unknown Injury Person Count',
    '16 Year Old Driver Count',
    '17 Year Old Driver Count',
    '18 Year Old Driver Count',
    '19 Year Old Driver Count',
    '20 Year Old Driver Count',
    '50-64 Year Old Driver Count',
    '65-74 Year Old Driver Count',
    '75 Plus Year Old Driver Count',
    'Unbelted Occupant Count',
    'Unbelted Death Count',
    'Unbelted Suspected Serious Injury Count',
    'Belted Death Count',
    'Belted Suspected Serious Injury Count',
    'Motorcycle Death Count',
    'Motorcycle Suspected Serious Injury Count',
    'Bicycle Death Count',
    'Bicycle Suspected Serious Injury Count',
    'Pedestrian Count',
    'Pedestrian Death Count',
    'Pedestrian Suspected Serious Injury Count',
    'Maximum Severity Level',
    'Commercial Vehicle Count',
    'Latitude (Decimal)',
    'Longitude (Decimal)',
    'Roadway Surface Type',
    'Interstate',
    'State Road',
    'Local Road Only',
    'Turnpike',
    'Wet Road',
    'Snow Slush Road',
    'Icy Road',
    'Sudden Deer',
    'Shoulder Related',
    'Rear End',
    'Head On / Opposite Direction Side Swipe',
    'Hit Fixed Object',
    'Single Vehicle Run Off Road',
    'Work Zone',
    # 'Property Damage Only',
    # 'Injury or Fatal',
    # 'Fatal or Suspected Serious Injury',
    # 'Injury',
    # 'Fatal',
    'Non Intersection',
    'Intersection',
    'Signalized Intersection',
    'Stop Controlled Intersection',
    'Unsignalized Intersection',
    'School Bus',
    'School Zone',
    'Hit Deer',
    'Hit Tree / Shrub',
    'Hit Embankment',
    'Hit Pole',
    'Hit Guide Rail',
    'Hit Guide Rail End',
    'Hit Barrier',
    'Hit Bridge',
    'Overturned',
    'Motorcycle',
    'Bicycle',
    'Heavy Truck Related',
    'Vehicle Failure',
    'Train / Trolley',
    'Phantom Vehicle',
    'Alcohol Related',
    'Drinking Driver',
    'Underage Drinking Driver',
    'Unlicensed',
    'Distracted',
    'Cell Phone',
    'No Clearance',
    'Running Red Light',
    'Tailgating',
    'Cross Median',
    'Curved Road',
    'Curve Driver Error',
    'Limit 65 MPH',
    'Speeding',
    'Speeding Related',
    'Aggressive Driving',
    'Fatigue / Asleep',
    'Driver 16 Years Old',
    'Driver 17 Years Old',
    'Driver 18 Years Old',
    'Driver 19 Years Old',
    'Driver 20 Years Old',
    'Driver 50 - 64 Years Old',
    'Driver 65 - 74 Years Old',
    'Driver 75 Plus',
    'Unbelted',
    'Pedestrian',
    'Commercial Vehicle',
    'PSP Reported',
    'NHTSA Aggressive Driving',
    'Deer Related',
    'Illumination Dark',
    'Running Stop Sign',
    'Train',
    'Trolley',
    'Hit Parked Vehicle',
    'Fire In Vehicle',
    'Vehicle Towed',
    'Hazardous Truck',
    'Suspected Serious Injury',
    'Suspected Minor Injury',
    'Possible Injury',
    'Motorcycle Drinking Driver',
    'Drug Related',
    'Illegal Drug Related',
    'School Bus Unit',
    'Drugged Driver',
    'Impaired Driver'
    ]

crash = crash[keepVar]

convertBoolean = [
    'School Bus Indicator',
    'School Zone Indicator',
    'Interstate',
    'State Road',
    'Local Road Only',
    'Turnpike',
    'Wet Road',
    'Snow Slush Road',
    'Icy Road',
    'Sudden Deer',
    'Shoulder Related',
    'Rear End',
    'Head On / Opposite Direction Side Swipe',
    'Hit Fixed Object',
    'Single Vehicle Run Off Road',
    'Work Zone',
    'Property Damage Only',
    'Injury or Fatal',
    'Fatal or Suspected Serious Injury',
    'Injury',
    'Fatal',
    'Non Intersection',
    'Intersection',
    'Signalized Intersection',
    'Stop Controlled Intersection',
    'Unsignalized Intersection',
    'School Bus',
    'School Zone',
    'Hit Deer',
    'Hit Tree / Shrub',
    'Hit Embankment',
    'Hit Pole',
    'Hit Guide Rail',
    'Hit Guide Rail End',
    'Hit Barrier',
    'Hit Bridge',
    'Overturned',
    'Motorcycle',
    'Bicycle',
    'Heavy Truck Related',
    'Vehicle Failure',
    'Train / Trolley',
    'Phantom Vehicle',
    'Alcohol Related',
    'Drinking Driver',
    'Underage Drinking Driver',
    'Unlicensed',
    'Distracted',
    'Cell Phone',
    'No Clearance',
    'Running Red Light',
    'Tailgating',
    'Cross Median',
    'Curved Road',
    'Curve Driver Error',
    'Limit 65 MPH',
    'Speeding',
    'Speeding Related',
    'Aggressive Driving',
    'Fatigue / Asleep',
    'Driver 16 Years Old',
    'Driver 17 Years Old',
    'Driver 18 Years Old',
    'Driver 19 Years Old',
    'Driver 20 Years Old',
    'Driver 50 - 64 Years Old',
    'Driver 65 - 74 Years Old',
    'Driver 75 Plus',
    'Unbelted',
    'Pedestrian',
    'Commercial Vehicle',
    'PSP Reported',
    'NHTSA Aggressive Driving',
    'Deer Related',
    'Illumination Dark',
    'Running Stop Sign',
    'Train',
    'Trolley',
    'Hit Parked Vehicle',
    'Fire In Vehicle',
    'Vehicle Towed',
    'Hazardous Truck',
    'Suspected Serious Injury',
    'Suspected Minor Injury',
    'Possible Injury',
    'Motorcycle Drinking Driver',
    'Drug Related',
    'Illegal Drug Related',
    'School Bus Unit',
    'Drugged Driver',
    'Impaired Driver',
    'Lane Closed'
]

for col in convertBoolean:
    if col in keepVar:
        crash[col] = crash[col].map({'Yes': 1, 'No': 0})
        crash[col] = crash[col].fillna(0)

for col in ['Traffic Detour',
            'Workers Present',
            'Work Zone Close/Detour',
            'Work Zone Flagger',
            'Work Zone Law Officer',
            'Work Zone Closure',
            'Work Zone Moving',
            'Work Zone Other',
            'Work Zone Shoulder/Median']:
    if col in keepVar:
        crash[col] = crash[col].map({'y': 1, 'Y': 1, 'n': 0, 'N': 0})
        crash[col] = crash[col].fillna(0)

convertCategorical = [
    # 'County Code',
    # 'Municipality Code',
    # 'District Code',
    # 'Police Agency Code',
    'Crash Month',
    'Day of Week',
    'Illumination',
    'Weather',
    'Road Condition',
    'Collision Type',
    'Work Zone Type',
    'Work Zone Location',
    'Intersection Type',
    'Traffic Control Device',
    'Location Type',
    'Urban / Rural',
    'Lane Closure Direction',
    'Notify Highway Maintenance',
    'Special Jurisdiction',
    'Traffic Control Device Function',
    'Roadway Surface Type'
]

convertCategorical = [x for x in convertCategorical if x in keepVar]

crashCategorical = pd.get_dummies(crash[convertCategorical], dummy_na=True)
crash = pd.concat([crash.drop(convertCategorical, axis=1), crashCategorical], axis=1)

# Fix dangling cases
crash['Estimated Hours Closed'] = crash['Estimated Hours Closed'].replace(r'\s+', 0, regex=True)
crash['Estimated Hours Closed'] = pd.to_numeric(crash['Estimated Hours Closed'])
crash['Construction Zone Speed Limit'] = crash['Construction Zone Speed Limit'].apply(lambda x: 1 if x > 0 else 0)

######################################################################################################################################################
# Summarize
######################################################################################################################################################

# muni = pd.read_csv('Municipality_Boundary.csv')
# muni['Municipality Code'] = muni['FIPS_MUN_C']
# crash = pd.merge(crash, muni[['Municipality Code', 'FIPS_STATE']], on='Municipality Code', how='outer')
# crash = crash.drop(['FIPS_STATE', 'County Code', 'District Code', 'Police Agency Code', 'Crash Record Number'], axis=1).fillna(0)
# crash_muni = crash.groupby('Municipality Code').sum().reset_index()
# print(crash_muni)
# crash_muni.to_csv('crash_muni.csv')

######################################################################################################################################################
# First Pass
######################################################################################################################################################

# Drop any accidents without lat/lon
crash = crash.dropna(axis=0)

X = crash.drop(['Crash Record Number', 'Fatality Count', 'Latitude (Decimal)', 'Longitude (Decimal)',
                'Municipality Code', 'County Code', 'District Code', 'Police Agency Code'], axis=1).as_matrix()
coord = crash[['Latitude (Decimal)', 'Longitude (Decimal)']].as_matrix()
y = crash['Fatality Count'].as_matrix()

X_train, X_test, coord_train, coord_test, y_train, y_test = model_selection.train_test_split(X, coord, y, test_size=0.1, random_state=100)

######################################################################################################################################################
# Models
######################################################################################################################################################

# ols = linear_model.LinearRegression()
# ridge = linear_model.Ridge()
# grad = linear_model.SGDRegressor()
# nb = naive_bayes.MultinomialNB()
# gp = gaussian_process.GaussianProcessRegressor()

# models = [ols, ridge, grad, nb, gp]

# print('=' * 150)
# print('=' * 150)
# print('Simple Regression')
# print('=' * 150)

# for mod in models:
#     print('=' * 150)
#     print('regression model:', mod.__class__.__name__)
#     mod.fit(X_train, y_train)
#     print('MSE: ', metrics.mean_squared_error(y_test, mod.predict(X_test)))
#     print('R^2: ', metrics.r2_score(y_test, mod.predict(X_test)))

######################################################################################################################################################
# Spatial Autocorrelation
######################################################################################################################################################

ols = linear_model.LinearRegression()
ridge = linear_model.Ridge()
grad = linear_model.SGDRegressor()
gp = gaussian_process.GaussianProcessRegressor()

models = [ols, ridge, grad, gp]

print(coord)

print('=' * 150)
print('=' * 150)
print('Kriging Regression')
print('=' * 150)

for mod in models:
    print('=' * 150)
    print('regression model:', mod.__class__.__name__)
    modRK = RegressionKriging(regression_model=mod, n_closest_points=20)
    modRK.fit(X_train, coord_train, y_train)
    print('Regression Rsquared: ', modRK.regression_model.score(X_test, y_test))
    print('RK Rsquared: ', modRK.score(X_test, coord_test, y_test))
