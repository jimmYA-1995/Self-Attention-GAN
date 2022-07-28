import os
import lmdb
from PIL import Image
import tempfile


db_path = "/root/notebooks/data/bridge_train_lmdb"

def _export_mdb_images(db_path, out_dir=None, flat=True, limit=-1, size=256):
    out_dir = out_dir
    env = lmdb.open(
        db_path, map_size=1099511627776,
        max_readers=1000, readonly=True
    )
    count = 0
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        for key, val in cursor:
            key = str(key, 'utf-8')
            # decide image out directory
            if not flat:
                image_out_dir = os.path.join(out_dir, '/'.join(key[:6]))
            else:
                image_out_dir = out_dir

            # create the directory if an image out directory doesn't exist
            if not os.path.exists(image_out_dir):
                os.makedirs(image_out_dir)

            with tempfile.NamedTemporaryFile('wb') as temp:
                temp.write(val)
                temp.flush()
                temp.seek(0)
                image_out_path = os.path.join(image_out_dir, key + '.jpg')
                Image.open(temp.name).resize((size, size)).save(image_out_path)
            count += 1
            if count == limit:
                break
            if count % 1000 == 0:
                print('Finished', count, 'images')

if __name__=="__main__":
    print("start")
    out_dir = os.path.join(db_path, "data")
    _export_mdb_images(db_path, out_dir)