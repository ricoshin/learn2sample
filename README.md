# learn2sample
Learning how to feed instance/classes/tasks efficiently using meta-learning framework

## Dependencies(updating..):

+ Python 3.3+.
+ PyTorch 1.0+.
+ torchviz (optional)

## How to run:

Test code (referring to ./gin/test.gin):
```
python main.py --gin test
```

With data & model parallelization:
```
python main.py --gin test --parallel
```

Note that gin file has to be specified.
Result has directories will be made at ./result/{gin_file_path}
