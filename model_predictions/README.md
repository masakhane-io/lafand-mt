### Model names

Insert the correct src and tgt language

- afribart_{src}_{tgt}_news.txt --- [afrimbart](https://huggingface.co/masakhane/afri-mbart50) fine-tuned on news
- afribyt5_{src}_{tgt}_news.txt --- [afribyt5](https://huggingface.co/masakhane/afri-byt5-base) fine-tuned on news
- afrimt5_{src}_{tgt}_news.txt --- [afrimt5](https://huggingface.co/masakhane/afri-mt5-base) fine-tuned on news
- mbart50_{src}_{tgt}_news.txt --- [mbart50](https://huggingface.co/facebook/mbart-large-50) fine-tuned on news
- byt5_{src}_{tgt}_news.txt --- [byt5](https://huggingface.co/google/byt5-base) fine-tuned on news
- mt5_{src}_{tgt}_news.txt --- [MT5](https://huggingface.co/google/mt5-base) fine-tuned on news
- m2m100_{src}_{tgt}_news.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) fine-tuned on news
- m2m100_{src}_{tgt}_rel.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) fine-tuned on religious corpus (REL) either JW300 or bible
- m2m100_{src}_{tgt}_rel_news.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) fine-tuned on aggregation of REL and news
- m2m100_{src}_{tgt}_rel_news_newsft.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) fine-tuned on aggregation of REL and news, followed by fine-tuning on news
- m2m100_{src}_{tgt}_rel_newsft.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) fine-tuned on REL, followed by fine-tuning on news
- m2m100_{src}_{tgt}_afro.txt  --- [M2M100](https://huggingface.co/facebook/m2m100_418M) multilingual fine-tuning on all french-centric or english centric corpus. Same as [FRENCH](https://huggingface.co/masakhane/m2m100_418M-FR-NEWS) or [ENGLISH](https://huggingface.co/masakhane/m2m100_418M-EN-NEWS)

The Test sets are also avaliable there
- test.src
- test.tgt

The transformer baselines follows the same pattern.
