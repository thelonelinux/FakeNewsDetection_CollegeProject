import csv

with open("/home/vicky/Desktop/datasets.txt") as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('/home/vicky/Desktop/datasets.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('post_id', 'post_text','user_id','image_id(s)', 'username','timestamp','label'))
        writer.writerows(lines)