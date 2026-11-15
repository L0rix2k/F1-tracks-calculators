haiii , this is the main core of the calculator, you need to have py 3.10 or newer (i use 3.13.17 ) then run:
pip install opencv-python numpy scipy scikit-image , after go to your Desktop, right-click on an empty spot, and choose New → Folder.
Name it F1track and press Enter.
Move the track_calc.py file into it ( check your Downloads folder if you can’t find it).
Put your top-down screenshot in the same folder and rename it to track.png. To rename, right-click the file, choose Rename, and type track.png.
Double-click the folder to open it, click the address bar at the top, type cmd, and press Enter.
then run cd Desktop\F1track and then dir for checking there is anything then:
python track_calc.py track.png --mode asphalt --crop 700,200,620,400 --real-length-m 9370 . easyyyyy
