#!/bin/bash

cd /home/dwemer/minime-t5/
python3 -m venv /home/dwemer/python-minime
source /home/dwemer/python-minime/bin/activate

echo "This will take a while so please wait."
pip install -r requirements.txt

./conf.sh




