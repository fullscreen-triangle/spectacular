FILE=$1
convert Screensh*.png -resize 480x480 new.png
convert -size 480x480 xc:#f9fafb  new.png  -gravity center -composite output.png
mv output.png $FILE
rm Screensh*.png new.png
