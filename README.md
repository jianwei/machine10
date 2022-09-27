sudo chmod -R 777 /dev/ttyACM0

python3 service.py 

python3 track.py --source 0 --yolo-weights ./weights/best.pt --camera_device 0
python3 track.py --source 0 --yolo-weights ./weights/best.pt --camera_device 2
