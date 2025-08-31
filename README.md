# INF2008

## requirements

- python version 3.12.7
- install ```pip install huggingface_hub nibabel```

## running with poetry

- ```poetry install --with dev```
- activate shell with ```poetry shell``` 

## commands

list subsets available at ctspine1k

```
spine download ctspine1k --list-subsets
```

## list 

download only head and neck scans

```
spine download ctspine1k --subset HNSCC-3DCT-RT --output-dir data/CTSpine1K
```