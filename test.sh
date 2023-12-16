# for i in {1..5}
# do
#   cls_thred=$(awk -v i="$i" 'BEGIN{print i*0.1}')
#   echo "cls_thred= $cls_thred"
#   python test.py -cls_thred $cls_thred
# done

python test.py -cls_thred 0.1
python test.py -cls_thred 0.2
python test.py -cls_thred 0.3
python test.py -cls_thred 0.4
python test.py -cls_thred 0.5