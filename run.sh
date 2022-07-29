for model in ./output_model/*;
do
    echo ${model} start
    python ./btnk2wav.py ${model} 1 ./output_wav/
done