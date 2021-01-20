from datetime import datetime
import time
import MetaTrader5 as mt5
import pandas as pd
from IPython.display import clear_output
import pickle
import re


# import pytz module for working with time zone
import pytz
# set time zone to UTC
timezone = pytz.timezone("Etc/UTC")


import six
import sys
from copy import deepcopy
sys.modules['sklearn.externals.six'] = six
from id3 import Id3Estimator
from sklearn.metrics import classification_report


if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

account=5330047

symbol=sys.argv[1]
TIMFRAME=sys.argv[2]

Start_range=3
end_range=50
NUMBER_OF_TEST_per_data=50
output_file=symbol+'-'+TIMFRAME

timeframe={
      "H":mt5.TIMEFRAME_H1,
    "H12":mt5.TIMEFRAME_H12,
      "D":mt5.TIMEFRAME_D1,
      "W":mt5.TIMEFRAME_W1,
      "MN":mt5.TIMEFRAME_MN1,
    }

authorized=mt5.login(account) 
if authorized:    
    account_info=mt5.account_info()
    terminal_info=mt5.terminal_info()
    

    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    utc_from = datetime(2009, 8, 26, 0, 0, 0, 0, tzinfo=timezone)
    
    utc_to   = datetime.today()
    rates  = pd.DataFrame(mt5.copy_rates_range(symbol, timeframe[TIMFRAME], utc_from, utc_to))

# shut down connection to the MetaTrader 5 terminal
mt5.shutdown()
rates.head()
print('mamad salam')
# candel recognition
df=rates.copy()
for i in range(2,df.shape[0]-1):
    current = df.iloc[i]
    prev = df.iloc[i-1]
    prev_2 = df.iloc[i-2]
    realbody = abs(current['open'] - current['close'])
    candle_range = current['high'] - current['low']
    
    idx = df.index[i]
    
    #label
    next_bar = df.iloc[i+1]
    df.loc[idx,'label'] = current['close'] < next_bar['close'] 
    
    # Bullish swing
    df.loc[idx,'Bullish swing'] = current['low'] > prev['low'] and prev['low'] < prev_2['low']

    # Bearish swing
    df.loc[idx,'Bearish swing'] = current['high'] < prev['high'] and prev['high'] > prev_2['high']

    # Bullish pinbar
    df.loc[idx,'Bullish pinbar'] = realbody <= candle_range/3 and  min(current['open'], current['close']) > (current['high'] + current['low'])/2 and current['low'] < prev['low']

    # Bearish pinbar
    df.loc[idx,'Bearish pinbar'] = realbody <= candle_range/3 and max(current['open'] , current['close']) < (current['high'] + current['low'])/2 and current['high'] > prev['high']

    # Inside bar
    df.loc[idx,'Inside bar'] = current['high'] < prev['high'] and current['low'] > prev['low']

    # Outside bar
    df.loc[idx,'Outside bar'] = current['high'] > prev['high'] and current['low'] < prev['low']

    # Bullish engulfing
    df.loc[idx,'Bullish engulfing'] = current['high'] > prev['high'] and current['low'] < prev['low'] and realbody >= 0.8 * candle_range and current['close'] > current['open']

    # Bearish engulfing
    df.loc[idx,'Bearish engulfing'] = current['high'] > prev['high'] and current['low'] < prev['low'] and realbody >= 0.8 * candle_range and current['close'] < current['open']
df.fillna(False, inplace=True)
x=[ 'label','spread' ,'tick_volume','Bullish swing', 'Bearish swing', 'Bullish pinbar', 'Bearish pinbar', 'Inside bar', 'Outside bar', 'Bullish engulfing', 'Bearish engulfing']
dfRaw=df[x]

dfRaw.to_csv('Models/'+output_file+'.csv',index=False)

t1=time.time()
n,k=0,len(range(Start_range,end_range))*NUMBER_OF_TEST_per_data

data=dfRaw.drop('label',1)
label=dfRaw.label
Result=list()

for Max_depth in range(Start_range,end_range):
    estimator = Id3Estimator(max_depth=Max_depth, min_samples_split=1, prune=True,
                    gain_ratio=True, min_entropy_decrease=0, is_repeating=True)
    temp=list()
    for NT in range(NUMBER_OF_TEST_per_data):
        clear_output(wait=True)
        n=n+1
        print('Learning Process has ',round((n/k)*100,4),' % complete : from ',output_file)
        estimator.fit(data,label , check_input=True)
        l=classification_report(label, estimator.predict(data))
        ACC=int(re.findall("True       \d.\d+", l)[0].split('.')[1])/100
        temp.append((ACC,deepcopy(estimator)))
    temp_index=[(x[0],i) for i,x in enumerate(temp)]
    temp_index.sort()
    Acc=temp[temp_index[-1][1]][0]
    estimator=temp[temp_index[-1][1]][1]
    Result.append(deepcopy(estimator))
t2=time.time()
print(round((t2-t1)/60) ,'min has time for calculaton Model')

with open(f"Models/{output_file}.HGH", "wb") as fp:
    pickle.dump(Result, fp)
