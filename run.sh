#!/bin/bash
cd "${0%/*}"
echo $PWD
source venv_linux/bin/activate
python apstopvoutput.py --pvoutput --tibber >> log.log

