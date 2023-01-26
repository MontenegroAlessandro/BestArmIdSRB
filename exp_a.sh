# for i in 100 150 200 250 300 350 400 450 500 600 700 800 900 1000 1200 1400 1600 1800 2000 2400 2800 3200
for ((i=100; i<=250; i+=5))
do
  cd config
  python sens_ucb.py $i
  cd ..
  python exp_sens_ucb_srb.py
done