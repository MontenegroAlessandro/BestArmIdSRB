for i in 500 1000 2000 3000 4000 5000 7000 10000 15000 20000 30000
do
  cd config
  python imdb_config.py $i 0.05 0.25
  cd ..
  python imdb_exp.py
done