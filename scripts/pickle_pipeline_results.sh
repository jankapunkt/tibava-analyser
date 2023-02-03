FILES="/home/emb/projects/tibava/data/media/**/*.yml"
ROOT_DIR="/home/emb/projects/tibava/data/media"

for f in $FILES
do
    echo "#### docker compose exec analyser python3 /app/analyser/scripts/pickle_pipeline_results.py -p /media/${f#$ROOT_DIR} -o /media"
    docker compose exec analyser python3 /app/analyser/scripts/pickle_pipeline_results.py -p /media/${f#$ROOT_DIR} -o /media
done
