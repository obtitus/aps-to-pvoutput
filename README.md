# aps-to-pvoutput
Upload data from AP Systems solar inverters (APS) automatically to PV Output

Extension of work done by https://github.com/willemstoker/aps-to-pvoutput but changed to work on a per-day basis. Uploading only full days. Intended to be run daily and not every 5-10 minutes.
Also added:
* openweather.org
* tibber.com (my power company, invite link https://invite.tibber.com/6a001fe9)

## Install
```
virtualenv venv;
pip install -r requirments.txt
```

## First use
before using first time, copy `my_api_keys_template.py` to `my_api_keys.py` and fill out the required information.

## Usage
see
```
python apstopvoutput.py --help
```
for usage. To automate the run, use either cron or launchd.

Tips, for mac users, use the supplied `local.aps-to-pvoutput.plist` and launchd. For a graphical interface, use [LaunchControl](https://www.soma-zone.com/LaunchControl/)
