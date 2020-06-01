#!/bin/bash
cd "${0%/*}"
source venv/bin/activate
python apstopvoutput.py --pvoutput --tibber
