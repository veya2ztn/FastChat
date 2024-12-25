
declare -a pids
LOG_FILE=".logging"

nohup python3 -m fastchat.serve.controller  > $LOG_FILE/controller.log 2>&1 &
pids[0]=$!

CUDA_VISIBLE_DEVICES=0 nohup python3 -m fastchat.serve.model_worker --model-path /home/niubility2/pretrained_models/PixArt-alpha/t5-v1_1-xxl/ --dtype float32  > $LOG_FILE/embedding.log 2>&1 &
pids[1]=$!

#### ---- 想开几个开几个
# CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path /home/niubility2/pretrained_models/PixArt-alpha/t5-v1_1-xxl/ --dtype float32
# CUDA_VISIBLE_DEVICES=2 python3 -m fastchat.serve.model_worker --model-path /home/niubility2/pretrained_models/PixArt-alpha/t5-v1_1-xxl/ --dtype float32
# CUDA_VISIBLE_DEVICES=3 python3 -m fastchat.serve.model_worker --model-path /home/niubility2/pretrained_models/PixArt-alpha/t5-v1_1-xxl/ --dtype float32
# CUDA_VISIBLE_DEVICES=4 python3 -m fastchat.serve.model_worker --model-path /home/niubility2/pretrained_models/PixArt-alpha/t5-v1_1-xxl/ --dtype float32

### use follow code test the ollama serve is setup correctly, you may create a is_ollama_build function
### curl http://$OLLAMA_HOST/api/embeddings -d '{"model": "jina/jina-embeddings-v2-base-en","prompt": "Llamas are members of the camelid family"}'
### setup a loop with timelimit that detect the status for ollama building via is_ollama_build function
nohup python3 -m fastchat.serve.openai_api_server --host localhost --port 8000 > $LOG_FILE/api_server.log 2>&1 &
pids[2]=$!

STARTTIME=$(date +%s)
while true; do
    curl http://localhost:8000/v1/embeddings -H "Content-Type: application/json" -d '{"model": "t5-v1_1-xxl","input": "Llamas are members of the camelid family"}'
    ### still in loop if we receive {"object":"error","message":"Only  allowed now, your model t5-v1_1-xxl","code":4030}
    if [ $? -eq 0 ]; then

        break
    fi
    echo "[$HOSTNAME][`date`] -- Embedding is not ready, check log file [$LOG_FILE/embedding.log] , wait 5s"
    sleep 5
    NOWTIME=$(date +%s)
    if [ $(($NOWTIME - $STARTTIME)) -gt 600 ]; then
        echo "[$HOSTNAME][`date`] -- Embedding is TIME OUT, check log file [$LOG_FILE/embedding.log] , wait 5s"
        exit 1
    fi
done
### 全部开起来之后

python callapi.py
### 理论上应该全是0