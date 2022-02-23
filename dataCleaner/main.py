import pandas as pd

data = pd.read_csv('cleanData.csv')
data.drop(['unitid',
           'IC2018_AY.In-district average tuition for full-time undergraduates',
          'IC2018_AY.Out-of-state average tuition for full-time undergraduates'],
          axis=1).to_csv('cleanData1.csv')
