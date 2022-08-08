SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

# download test file
sudo docker compose exec analyser wget "https://tib.eu/cloud/s/ddsr9M9LD3NfGyc/download/xg_mascara.mp4" -O /media/test.mp4

# test each test_script.py
FILES="$SCRIPT_DIR/*.py"
for f in $FILES
do
    fname=$(basename "$f")
    echo "#### sudo docker-compose exec analyser python3 /app/analyser/test/$fname -v"
    sudo docker compose exec analyser python3 /app/analyser/test/$fname -v
done