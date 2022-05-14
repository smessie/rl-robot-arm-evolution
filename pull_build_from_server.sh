SERVER_SSH_CONFIG=gorilla

scp scp://$SERVER_SSH_CONFIG/simenv.tar.gz .

rm -rf build/
tar -xvf simenv.tar.gz

