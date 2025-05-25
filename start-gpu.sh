source /home/dwemer/python-minime/bin/activate

cd /home/dwemer/minime-t5/

uvicorn main:app --port 8082 &>log.txt&


