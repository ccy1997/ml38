def transform_ocean_proximity(housing):
    housing['1h_ocean'] = [1 if i=='<1H OCEAN' else 0 for i in housing.ocean_proximity.values]
    housing['island'] = [1 if i=='ISLAND' else 0 for i in housing.ocean_proximity.values]
    housing['inland'] = [1 if i=='INLAND' else 0 for i in housing.ocean_proximity.values]
    housing['near_ocean'] = [1 if i=='NEAR OCEAN' else 0 for i in housing.ocean_proximity.values]
    housing['near_bay'] = [1 if i=='NEAR BAY' else 0 for i in housing.ocean_proximity.values]
    housing.drop(columns=['ocean_proximity'], inplace=True)
