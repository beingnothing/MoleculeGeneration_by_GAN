var=./dude
dirlist=`ls ./dude`
name=actives_final.sdf.gz
for subdir in $dirlist
do
gzip -d $var/$subdir/$name
done