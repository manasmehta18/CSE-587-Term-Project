mkdir setup_env
cd setup_env

# install pytorch
# see https://download.pytorch.org/whl/torch_stable.html
echo "install torch==1.8.0+cu111\n"
wget https://download.pytorch.org/whl/cu111/torch-1.8.0%2Bcu111-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu111/torchvision-0.9.0%2Bcu111-cp38-cp38-linux_x86_64.whl
pip install torch-1.8.0+cu111-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.9.0+cu111-cp38-cp38-linux_x86_64.whl

# install torch-geometric
# see https://data.pyg.org/whl/index.html
echo "install torch_geometric==2.0.2\n"
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
wget https://data.pyg.org/whl/torch-1.8.0%2Bcu111/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.10-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl

pip install torch_geometric==2.0.2

# remove 
cd ../
rm -rf setup_env