filename="tags.csv.20210301.poly"
new_size=400
dest_dir="data/annotated_imgs/"
#classes=["mummification","fly","maggots","scavenging","eggs"]
classes=["mummifi","fl","maggot","scav","egg"] #to include all cases

python3 fix.py $filename $classes

sed -i 's/"/""/g' $filename"_fixed"
sed -r -i  's/\),\[/\),\"\[/g' $filename"_fixed"
sed -r -i 's/],sara/]",sara/g' $filename"_fixed"

python3 generate_annotated_images.py $filename $classes $new_size $dest_dir

python3 color_imgs_by_dir.py $dest_dir


#python3 pair_generator.py
echo "\n Generating the odgt files and checking the number of each class in the train, val, and test split"
bash check_distribution.sh tags.csv.20210301.poly_fixed

