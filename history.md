1463  exit
 1464  screen -r
 1465  screen
 1466  exit
 1467  ls
 1468  screen -r
 1469  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/dslr/ --ip 0.0.0.0 --port 16006 --depth_ratio 1
 1470  conda activate ml3d
 1471  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/dslr/ --ip 0.0.0.0 --port 16006 --depth_ratio 1
 1472  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/dslr/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --eval
 1473  screen -r
 1474  conda activate ml3d
 1475  pip install tensorboard
 1476  conda activate ml3d
 1477  ls
 1478  cd ..
 1479  pip install -r requirements.txt
 1480  cd iphone
 1481  python -m prepare_iphone_data configs/prepare_iphone_data.yml
 1482  cd ..
 1483  python -m iphone.prepare_iphone_data configs/prepare_iphone_data.yml
 1484  python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
 1485  ls /cluster/51/koubaa/data/scannet++/data/data/07f5b601ee
 1486  ls /cluster/51/koubaa/data/scannet++/data/data/07f5b601ee/iphone
 1487  python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
 1488  conda activate ml3d
 1489  cd ..
 1490  ls
 1491  cd 2dGScode
 1492  ls
 1493  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/dslr/ --ip 0.0.0.0 --port 16006 --depth_ratio 1
 1494  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/dslr/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/dslr/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --image_subdir 'resized_undistorted_images'
 1495  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --image_subdir 'rgb''
 1496  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --image_subdir 'rgb''
 1497  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --image_subdir 'rgb'
 1498  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' 
 1499  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json'
 1500  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8
 1501  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --eval
 1502  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval
 1503  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1504  exit
 1505  git clone git@github.com:scannetpp/scannetpp.git
 1506  cd scannetpp/iphone
 1507  conda activate ml3d
 1508  /cluster/51/koubaa/data/scannet++
 1509  ls/cluster/51/koubaa/data/scannet++
 1510  ls /cluster/51/koubaa/data/scannet++
 1511  ls /cluster/51/koubaa/data/scannet++/data
 1512  python -m python -m iphone.prepare_iphone_data iphone/configs/prepare_iphone_data.yml
 1513  python -m prepare_iphone_data configs/prepare_iphone_data.yml
 1514  conda deactivate
 1515  salloc --gpus 1
 1516  screen -r
 1517  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1518  conda activate ml3d
 1519  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1520  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1521  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 30000
 1522  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval 
 1523  salloc --gpus 1
 1524  sinfo
 1525  salloc --gpus 1
 1526  ls
 1527  screen -li
 1528  salloc --gpus 1
 1529  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval
 1530  conda activate ml3d
 1531  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval
 1532  python 2dGScode/train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_image' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms undistorted.json'  --use_exposure_optimization --eval
 1533  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_image' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms undistorted.json'  --use_exposure_optimization --eval
 1534  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_image' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms undistorted.json'  --use_exposure_optimization --eval
 1535  /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/../dslr/nerfstudio/transforms undistorted.json
 1536  /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/../dslr/nerfstudio/
 1537  /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/../dslr/nerfstudio/transforms_undistorted.json
 1538  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_image' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json'  --use_exposure_optimization --eval
 1539  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_images' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json'  --use_exposure_optimization --eval
 1540  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_images' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' --eval
 1541  python 2dGScode/train.py --source_path /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ --model_path /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --depth_ratio 1 --images 'rgb' --test_images '../dslr/resized_undistorted_images' --train_transforms_file 'nerfstudio/transforms.json' --test_transforms_file '../dslr/nerfstudio/transforms_undistorted.json' --use_exposure_optimization --eval
 1542  ls
 1543  history
 1544  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1545  /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/../dslr/nerfstudio/transforms undistorted.json
 1546  history
 1547  conda activate ml3d
 1548  /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/../dslr/nerfstudio/transforms undistorted.json
 1549  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1550  ls
 1551  cd ..
 1552  ls
 1553  cd /
 1554  ls
 1555  python train.py -s /cluster/51/koubaa/data/scannet++/data/0b031f3119/iphone/ -m /cluster/51/koubaa/data/output/scannet++/0b031f3119/iphone/ --ip 0.0.0.0 --port 16006 --depth_ratio 1 --images 'rgb' --transforms_file 'nerfstudio/transforms.json' --split 0.8 --use_exposure_optimization --eval --iterations 60000
 1556  cd cluster/51/koubaa/
 1557  ls
 1558  cd mahdi/
 1559  ls
 1560  cd 2DGaussianSplatting/
 1561  ls
 1562  history 100