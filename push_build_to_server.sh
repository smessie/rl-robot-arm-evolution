SERVER_SSH_CONFIG=gorilla

tar -czvf simenv.tar.gz build/

scp simenv.tar.gz scp://$SERVER_SSH_CONFIG

ssh -- $SERVER_SSH_CONFIG "rm -vrf build/; tar -xvf simenv.tar.gz"

