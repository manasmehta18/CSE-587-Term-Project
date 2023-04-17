cd ./data

echo '\n*** launch new folder "RECCON/beta_BIO" ***\n'
python process_beta_BIO.py
echo '\n*** launch new folder "RECCON/alpha_BIO" ***\n'
python process_alpha_BIO.py
echo '\n*** launch new folder "RECCON_BIO" ***\n'
python process_BIO.py