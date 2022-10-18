#!/bin/bash

ad=Section001

for id in $(ls $ad);
do
    echo "id $id"
    stu=$ad/$id/01/
    unzip -o $ad/$id/01/*.zip -d $stu
    for dir in $stu/*/;
    do 
        echo "$dir"
        cat $dir/*.txt;
        make -C $dir ;
        cd $dir && run.sh -t ;
    done;
    cd ~
    echo $(pwd)
    printf "\n\t***************** $id *****************\n\n"
    read -p "Press Enter to continue" </dev/tty	
    clear;
done
