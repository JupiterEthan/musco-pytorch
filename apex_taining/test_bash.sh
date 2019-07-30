#!/bin/bash

wfs=(0.1 0.2 0.9)
lenwfs=${#wfs[@]}

echo $lenwfs
a=1

for (( i=0; i<$lenwfs; i++)) 
do 
            fact="${wfs[$i]}"
            if [[ $i == `expr $lenwfs - $a` ]]
            then
                echo 'Largest wf' $fact
            fi
            
            if [[ $i == 0 ]]
            then
                echo 'Smallest wf' $fact
            fi
done
