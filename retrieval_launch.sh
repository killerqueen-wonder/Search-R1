
# file_path=/the/path/you/save/corpus
# file_path=/mnt/nvme1n1/legal_LLM/dataset
# file_path=/share/home/panghuaiwen/legal_LLM/dataset
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            "$@"
