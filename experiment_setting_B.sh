for i in 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2400 2800 3200
do
  cd config
  python config_setting_B.py $i
  cd ..
  python experiment_setting_B.py
done