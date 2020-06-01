#!/usr/bin/python
import logging
logger = logging.getLogger(__name__)

import requests
import json
from datetime import date, datetime, timedelta
import os.path

# simple re-sample to every hour
import pandas as pd
# weather info
from pyowm.owm import OWM
from pyowm.commons.exceptions import APIRequestError

# Note: only needed for plotting
import pylab as plt
import matplotlib.dates as mdates

from my_api_keys import *

#enter a path and filename below, a file wil be create to save the last update datetime
LAST_UPDATE_FILE = "last_update_file.txt" # keep a record of latest date uploaded

#usually all below this point should not be modified
APSYSTEMS_URL = 'http://api.apsystemsema.com:8073/apsema/v1/ecu/getPowerInfo'
PVOUTPUT_URL = 'https://pvoutput.org/service/r2/addoutput.jsp'
TIBBER_URL = 'https://api.tibber.com/v1-beta/gql'

def read_date_from_file(LAST_UPDATE_FILE):
    with open(LAST_UPDATE_FILE, "r") as f:
        datestring = f.read().strip()

    return datetime.strptime(datestring, "%Y%m%d")

def write_date_to_file(LAST_UPDATE_FILE, date_str):
    '''
    date_str is assumed to be in format %Y%m%d
    '''
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_str)

def setupLogger(level=logging.DEBUG):
    logger.setLevel(level)
    l_handler = logging.StreamHandler()
    l_handler.setLevel(logging.DEBUG)

    l_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    l_handler.setFormatter(l_format)
    logger.propagate = False
    logger.addHandler(l_handler)

def getDateString():
    try:
        wanted_day = read_date_from_file(LAST_UPDATE_FILE) + timedelta(days=1)
    except FileNotFoundError: # first iteration
        wanted_day = datetime.strptime(FIRST_DATE, "%Y%m%d")
    
    if datetime.today().date() <= wanted_day.date():
        return '' # if today, stop
    else:
        return wanted_day.strftime("%Y%m%d")

def getDataFromAPS(wanted_day):
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    data = {
        'ecuId': ECU_ID,
        'filter': 'power',
        'date': wanted_date
    }

    logging.info('Requesting %s from %s...', data, APSYSTEMS_URL)
    response = requests.post(APSYSTEMS_URL, headers=headers, data=data)
    logging.debug(response)
    return response.json();

def getWeather(df_resampled, inplace=True):
    if not(inplace):
        df_resampled = pd.DataFrame(df_resampled)
    
    solar_panel_sunrise = df_resampled.index[0].timestamp()
    solar_panel_sunset  = df_resampled.index[-1].timestamp()
    
    # Retrieve current weather information (Requires the installation of pyowm. $ pip install pyowm)
    owm = OWM(OPENWEATHER) # a Python wrapper around the OpenWeatherMap API
    mgr = owm.weather_manager() # See: https://pyowm.readthedocs.io/en/latest/v3/code-recipes.html
    
    weather = mgr.one_call_history(lat=LAT, lon=LON, dt=round(solar_panel_sunrise))
    #weather.forecast_daily # not sure why this is empty, bug?
    df_resampled['Temperature [C]']  = 0 # initialize all rows
    df_resampled['Status']           = '' # initialize all rows
    df_resampled['Detailed status']  = '' # initialize all rows
    df_resampled['Cloud cover [%]']  = 0 # initialize all rows
    df_resampled['Wind speed [m/s]'] = 0 # initialize all rows
    df_resampled['Wind angle [deg]'] = 0 # initialize all rows
    
    for forecast in weather.forecast_hourly:
        f_time = forecast.reference_time()
        f_time_str = datetime.utcfromtimestamp(f_time).strftime('%H:%M')
        ix = df_resampled.index.strftime('%H:%M') == f_time_str # find corresponding row
        
        if any(ix):
            df_resampled.loc[ix, 'Temperature [C]'] = forecast.temperature(unit='celsius').get('temp', 0)
            df_resampled.loc[ix, 'Status']          = forecast.status # e.g. rain
            df_resampled.loc[ix, 'Detailed status'] = forecast.detailed_status # e.g. light rain
            df_resampled.loc[ix, 'Cloud cover [%]'] = forecast.clouds
            wind = forecast.wind()
            df_resampled.loc[ix, 'Wind speed [m/s]'] = wind['speed']
            df_resampled.loc[ix, 'Wind angle [deg]'] = wind['deg']

            #forecast.sunrise_time() # seems to be None, at least in June
            #forecast.sunset_time() # seems to be None, at least in June
        else:
            logger.debug('ignoring weather at %s' % f_time_str)
    
    return df_resampled

def sendUpdateToPVOutput(df, df_resampled, total_kwh, tibber_data=None):
    output_date = df_resampled.index[0].strftime('%Y%m%d')
    generated = total_kwh*1e3
    exported = ''
    peak_power = df['Power [W]'].max()
    peak_time = df.index[df['Power [W]'].argmax()].strftime('%H:%M')
    # note: order and format must match pvoutput api
    data = [output_date, generated, exported, peak_power, peak_time]

    condition = ''
    min_temp = ''
    max_temp = ''
    comment = ''
    
    if 'Temperature [C]' in df_resampled:
        min_temp = df_resampled['Temperature [C]'].min()
        max_temp = df_resampled['Temperature [C]'].max()

        status = df_resampled['Status'].value_counts().index[0] # median for string
        detailed_status = df_resampled['Detailed status'].value_counts().index[0] # median for string
        logger.debug('median condition %s:\n>%s', df_resampled['Detailed status'].value_counts(), detailed_status)

        cloud_cover_per = df_resampled['Cloud cover [%]'].median()

        condition_pvoutput = {'clear': 'Fine',
                              'clear sky': 'Fine', 
                              'few clouds': 'Partly Cloudy',
                              'scattered clouds': 'Mostly Cloudy',
                              'broken clouds': 'Cloudy',
                              'overcast clouds': 'Cloudy',
                              'mist': 'Cloudy',
                              'shower rain': 'Showers',
                              'rain': 'Showers',
                              'thunderstorm': 'Showers',
                              'snow': 'Snow'}

        if status.lower() == 'clouds':
            # switch status to detailed status to get more fine grained info on cloud cover
            status = detailed_status
            # add percentage to detailed status (used in comment)
            detailed_status += ' %g %%' % (cloud_cover_per)

        condition = condition_pvoutput.get(status.lower(), 'Not Sure') # try to map pvoutput weather and openweather api weather values
        comment = detailed_status
        logger.info('Weather = %s -> %s. detailed=%s' % (status, condition, comment))
        data.extend([condition, min_temp, max_temp, comment])

    if tibber_data is not None:
        # try to find correct date
        ix = tibber_data.index.strftime('%Y%m%d') == output_date
        if any(ix):
            import_peak = ''
            import_off_peak = ''
            import_shoulder = ''
            import_high_shoulder = ''

            consumption = 1e3*(tibber_data.loc[ix, 'Consumption [kWh]'].sum() + tibber_data.loc[ix, 'Production [kWh]'].sum())

            if len(data) == 5:
                # pad with empty data
                data.extend([condition, min_temp, max_temp, comment])
                logger.debug('No weather data, padding data %s', data)
                
            data.extend([import_peak, import_off_peak, import_shoulder, import_high_shoulder, consumption])
        else:
            logger.warning('Tibber data does not contain requested date %s', output_date)

    def my_float_to_str(s):
        try:
            return '%g' % s
        except TypeError:
            return s
    
    pvoutputdata = {
        'data': ','.join(map(my_float_to_str, data))
    }

    headerspv = {
        'X-Pvoutput-SystemId': PV_OUTPUT_SYSTEMID,
        'X-Pvoutput-Apikey': PV_OUTPUT_APIKEY
    }

    logger.info('Posting %s %s to %s', pvoutputdata, headerspv, PVOUTPUT_URL)
    r = requests.post(PVOUTPUT_URL, headers=headerspv, data=pvoutputdata)
    logger.info("Response: %s", r.text)
    r.raise_for_status()

def getTibberConsumption(last_n_days = 30):
    # there is some kind of support for before and after (date??) string, but I can't figure ut out, so just query the last 30 days and pick out the wanted date from that.
    query = """
    {
        viewer {
            homes {
                consumption(resolution: DAILY, last: %d) {
                    nodes {
                        from, to, cost, unitPrice, unitPriceVAT, consumption, consumptionUnit
                    }
                }
                production(resolution: DAILY, last: %d) {
                    nodes {
                        from, to, profit, unitPrice, unitPriceVAT, production, productionUnit
                    }
                }
            }
        }
    }
    """ % (last_n_days, last_n_days)

    # compact the query
    res = [line.strip() for line in query.split()]
    query_compact = ' '.join(res)
    
    payload = {
        "query": query_compact,
    }

    header = {
        "Authorization": "Bearer " + TIBBER_TOKEN,
    }

    logger.info('Requesting tibber data %s %s %s' % (payload, header, TIBBER_URL))
    r = requests.post(TIBBER_URL, headers=header, data=payload)
    #logger.info("Response: %s", r.text)
    r.raise_for_status()
    
    data = r.json()
    homes = data['data']['viewer']['homes']
    assert len(homes) == 1, 'Vopsy, expected single tibber home %s' % homes
    consumption = homes[0]['consumption']['nodes']
    production  = homes[0]['production']['nodes']
    assert len(consumption) == len(production), 'expected number of consumption and production records to be the same %d != %d' % (consumption, production)

    tibber_data = list()
    tibber_date = list()
    for ix in range(len(consumption)):
        c = consumption[ix]
        p = production[ix]
        assert c['from'] == p['from']
        assert c['consumptionUnit'] == 'kWh'
        assert p['productionUnit']  == 'kWh'

        tibber_date.append(c['from'])
        tibber_data.append({'Cost [NOK]': c['cost'],
                            'Profit [NOK]': p['profit'],
                            'Consumption [kWh]': c['consumption'],
                            'Production [kWh]': p['production']})

    dates = pd.to_datetime(tibber_date)
    df = pd.DataFrame(tibber_data, index=dates)
    
    return df
    
def plot_pv_data(df, df_resampled, title):
    fig, ax = plt.subplots()

    ax.plot(df.index, df['Power [W]'])
    ax.plot(df_resampled.index, df_resampled['Power [W]'], drawstyle="steps-post")

    ax.set_xlabel('Time')
    ax.set_ylabel('Power [W]')
    ax.set_title(title)
    # format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()

if __name__ == '__main__':
    setupLogger()

    import argparse

    parser = argparse.ArgumentParser(description='Downloads data from apsystem and uploads to pvoutput.org')
    parser.add_argument('--plot', action='store_true',
                        default=False,
                        help='Optionally plot data')
    parser.add_argument('--pvoutput', action='store_true',
                        default=False,
                        help='Optionally upload data to pvoutput.org')
    parser.add_argument('--max_days', '-d', default=30,
                        help='Specify the maximum number of days to process. Defaults to 30 days')
    
    args = parser.parse_args()

    tibber_data = getTibberConsumption()
    
    day_count = 0
    wanted_date = getDateString()
    while wanted_date != '' and day_count < int(args.max_days):
        rootdict = getDataFromAPS(wanted_date)
        print(rootdict.keys())
        print(rootdict['data'].keys())
        timesstring = rootdict.get("data").get("time")
        powersstring = rootdict.get("data").get("power")

        timelist = json.loads(timesstring)
        powerlist = json.loads(powersstring)

        # add the date:
        timelist = list(map(lambda x: wanted_date + ' ' + x, timelist))
        for ix in range(len(powerlist)):
            print(timelist[ix], powerlist[ix])

        # convert to pandas time
        timelist = pd.to_datetime(timelist)
        # create dataframe
        df = pd.DataFrame(powerlist, index=timelist, columns=['Power [W]'], dtype=int)
        
        # Resample to whole hours and calculate kWh
        df_resampled = df.resample('H').mean()
        total_kwh = df_resampled['Power [W]'].mean()*24/1e3 # not sure if this is right, should probably do a better numerical integration

        try:
            getWeather(df_resampled, inplace=True)
        except APIRequestError as e:
            logger.exception(e)

        if args.pvoutput:
            # upload to pvoutput
            sendUpdateToPVOutput(df, df_resampled, total_kwh, tibber_data)
            
        if args.plot:
            title = '%s. Total: %.0f [kWh]' % (wanted_date, total_kwh)
            plot_pv_data(df, df_resampled, title=title)
            
        # prep for next loop
        day_count += 1
        write_date_to_file(LAST_UPDATE_FILE, wanted_date)
        wanted_date = getDateString()
        
    if args.plot:
        plt.show()

    # i = len(timelist) - 1
    # count = 0;
    # while i >= 0 and count < MAX_NUMBER_HISTORY:
    #     timestringminutes = timelist[i][:-3] #get time and strip the seconds
    #     powerstring = powerlist[i] #get power

    #     currentUpdate = datetime.strptime(getDateStringOfToday()+ ' ' +timestringminutes, "%Y%m%d %H:%M")

    #     print(timestringminutes, powerstring)
    #     # if currentUpdate > latestUpdate:
    #     #     sendUpdateToPVOutput(timestringminutes, powerstring)
    #     # else:
    #     #     print("No update needed for: " + timestringminutes)

    #     if count == 0:
    #         writeLastUpdate(timestringminutes)

    #     i -= 1
    #     count += 1
