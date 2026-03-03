## Face Similarity Analysis System project (from manhtuanthusinh)

+ dùng conda tạo 1 môi trường python mới theo file environment.yml có sẵn :

    ```bash
    conda env create -f environment.yml'
    ```
+ vào file config.py chỉnh sửa :

    #### DEVICE (nếu dùng cpu thì 'cpu' còn gpu thì 'cuda:0')
    
    #### DATASET_PATH (địa chỉ tới folder dataset ảnh)

+ chạy hàm main1 để lấy output features.npy và labels.npy 
+ sau đó chạy correlation.py 