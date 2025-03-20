```bash
python -m distill_cli distill --config config.json --output-dir ./distilled_model
```


```bash
python -m distill_cli distill_cli --teacher teacher_model.pt --student small_model.pt --train-data train.pt --val-data val.pt --epochs 20 --batch-size 64 --lr 0.001 --temperature 3.0 --alpha 0.7 --compression 0.3 --output-dir ./distilled
```
