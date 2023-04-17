cd ./pretrained

echo 'downloading the spanbert-base-cased...'
cd language_model
wget https://dl.fbaipublicfiles.com/fairseq/models/spanbert_hf_base.tar.gz
tar -zxvf spanbert_hf_base.tar.gz -C spanbert-base-cased
rm spanbert_hf_base.tar.gz
cd ../word_embedding
echo 'downloading the glove...'
wget https://nlp.stanford.edu/data/glove.6B.zip
mkdir glove
unzip glove.6B.zip -d glove
mv glove/glove.6B.300d.txt ./
rm -r glove
rm glove.6B.zip