#!/bin/bash
# author: 
rm -r ./code/TSP.o
rm -r ./test
make

Temp_City_Num=$1
Total_Instance_Num=$2
threads=$3
Inst_Num_Per_Batch=$(($Total_Instance_Num/$threads))

for ((i=0;i<$threads;i++));do
{
	touch ./results/${Temp_City_Num}/result_${i}.txt
	./test $i ./results/${Temp_City_Num}/result_${i}.txt ./tsp${Temp_City_Num}_test_concorde.txt ${Temp_City_Num} ${Inst_Num_Per_Batch}
}&
done
wait

echo "Done."
