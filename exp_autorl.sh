clear
for ((i=20; i<=150; i+=10))
do
  # for j in 0.1 0.2 0.3 0.4 0.5
  for j in 0.2
  do
    #for z in 0.2 0.3 0.4 0.49
    for z in 0.25
      do
      cd config
      python autorl_config.py $i $j $z
      cd ..
      python autorl_exp.py
      done
  done
done