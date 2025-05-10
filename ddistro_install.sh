#!/bin/bash

cd /home/dwemer/minime-t5/
python3 -m venv /home/dwemer/python-minime
source /home/dwemer/python-minime/bin/activate

echo "Installing MiniMe-T5 and TXT2VEC..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
  echo "Pip install failed."
  read -p "Press Enter to attempt to continue, or Ctrl+C to exit."
fi

./conf.sh




