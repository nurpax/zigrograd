import os
import requests # pip install requests
import hashlib
import gzip

def download_file(url: str, out_path):
    response = requests.get(url)

    if response.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(response.content)
        print(f'downloaded {os.path.basename(out_path)}')
    else:
        print('Failed to download the file. Status code:', response.status_code)
        assert False, 'download failed'

def main():
    out_dir = 'data/mnist'
    os.makedirs(out_dir, exist_ok=True)

    files = [
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'),
    ]

    for filename, hash in files:
        out_file = os.path.join(out_dir, filename)
        if os.path.exists(out_file):
            with open(out_file, 'rb') as f:
                c = f.read()
                md5_hash = hashlib.md5()
                md5_hash.update(c)
                assert md5_hash.hexdigest() == hash
                print('file exists', out_file)
        else:
            download_file(f'http://yann.lecun.com/exdb/mnist/{filename}', out_file)

        with gzip.open(out_file, 'rb') as gz_file:
            out_extracted = out_file.replace('.gz', '')
            with open(out_extracted, 'wb') as f:
                f.write(gz_file.read())
                print(f'extracted to {out_extracted}')
    print('done')

if __name__ == '__main__':
    main()
