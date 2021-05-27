#if we are checking for only one file like sup_val.odgt: bash check_distribution.sh sup_val.odgt tags.csv.20210301.poly_fixed
# or if the files are added in the code: bash check_distribution.sh tags.csv.20210301.poly_fixed

filename=""
tags=""

function count () {
	cat $filename | cut -d ":" -f 2| cut -d "," -f 1 | rev | cut -d "/" -f 1 | rev | cut -d "\"" -f 1 |sort -u | while read line
	do 
		grep $line $tags >> labels
	done
	
	cat labels | awk -F "," '{print $(NF -3)}' | sort | uniq -c
	rm labels
}

if [ "$#" -eq 2 ]; then
        filename=$1
	tags=$2
	count
fi

if [ "$#" -eq 1 ]; then
	#python3 pair_generator.py
	files="data/sup_train.odgt data/sup_val.odgt data/sup_test.odgt"
	tags=$1
	for f in $files
	do
		filename=$f
		echo $filename
		count
	done
fi
		
