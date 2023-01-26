for i in 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2400 2800 3200
do
  for j in 0.001 0.005 0.01 0.05 0.1 0.5
  do
    cd config
    python config_setting_A.py $i $j
    cd ..
    python experiment_setting_A.py
  done
done