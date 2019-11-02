# learn2sample
Learning how to feed instance/classes/tasks efficiently using meta-learning framework

## Dependencies(updating..):

+ Python 3.3+.
+ PyTorch 1.0+.
+ gin (https://github.com/google/gin-config)
+ pandas
+ seaborn

## How to run:

Test code (referring to ./gin/test.gin):
```
python train.py --gin test
```

With data & model parallelization:
```
python train.py --gin test --parallel
```

For debugging:
```
python train.py --gin debug --volatile
```
The flag --volatile will skip all the file savings.

Note that gin file has to be specified.
Result has directories will be made at ./result/{gin_file_path}

SIGINT(^C) and SIGTSTOP(^Z) will be intercepted by utils.utils.SignalCatcher.

Both will be used for debugging or log control. Use SIGQUIT(^\) to terminate the process.
