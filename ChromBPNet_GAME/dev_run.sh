#!/bin/bash
# USAGE: ./dev_run.sh <matcher_ip> <matcher_port>
# Mounts the local ChromBPNet directory into the container so edited scripts
# are used directly without rebuilding the SIF.

# 1. Define the image path
CONTAINER_IMG="chrombpnet_predictor.sif"

# 2. Check that the image exists
if [ ! -f "$CONTAINER_IMG" ]; then
    echo "❌ Error: Could not find container at $CONTAINER_IMG"
    echo "   Make sure you are running this script from the 'ChromBPNet_GAME' directory."
    exit 1
fi

# 3. Require matcher arguments — ChromBPNet always needs the Matcher service
if [[ -z "$1" || -z "$2" ]]; then
    echo "❌ Error: Matcher IP and port are required."
    echo "   USAGE: ./dev_run.sh <matcher_ip> <matcher_port>"
    exit 1
fi


matcher_ip=$1
matcher_port=$2

# 4. Resolve predictor IP and a free port
pred_ip=$(hostname -I | awk '{print $2}')
pred_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

PY_ARGS="$pred_ip $pred_port $matcher_ip $matcher_port"

echo "=========================================================="
echo "🧪 STARTING DEV MODE: ${CONTAINER_IMG}"
echo "=========================================================="
echo "   Mapping host '$PWD' ---> Container '/ChromBPNet_GAME' (Working Dir)"
echo "   Predictor : http://$pred_ip:$pred_port"
echo "   Matcher   : http://$matcher_ip:$matcher_port"
echo "----------------------------------------------------------"

# 5. The Apptainer command
#    - Bind the entire project directory so all local edits are visible
#    - PYTHONPATH includes the project root (for chrombpnet package imports)
#      and the chrombpnet subdirectory (for internal chrombpnet imports)
apptainer exec --nv \
    --bind $PWD:/ChromBPNet_GAME \
    --pwd /ChromBPNet_GAME \
    --env PYTHONPATH="/ChromBPNet_GAME:/ChromBPNet_GAME/chrombpnet:$PYTHONPATH" \
    "$CONTAINER_IMG" \
    python3 /ChromBPNet_GAME/ChromBPNet_predictor_RestAPI.py $PY_ARGS