# TraceSim
TraceSim is a method for measuring stacktrace similarity.

We describe details of algorithm in our MaLTeSQuE 2020 paper "TraceSim: A Method for Calculating Stack Trace Similarity", which can be found here: https://arxiv.org/abs/2009.12590.

Also, we implement other well-known methods for comparison on our data.

### Usage
```bash
python main.py
    --stacktrace_dir [path_to_dir_with_stacktraces]
    --labels_path [path_to_labels] 
    --method tracesim
```

We assume that
- `stacktrace_dir` is a directory with json files. 
See example of file with stacktrace below.
- `labels_path` is a csv file with header `rid1,rid2,label`.
- `method` mean name of the method for measure similarity. 
One of the lerch, moroo, rebucket, cosine, levenshtein, brodie, prefix. 
Also, you can pass `all` for all methods comparison.

### Stacktrace example

```json
{
  "id": 223576912,
  "timestamp": 1.942130912798E12,
  "class": [
    "java.lang.ClassNotFoundException",
    "java.lang.RuntimeException"
  ],
  "frames": [
    [
      "java.lang.ClassLoader.loadClass",
      "java.lang.Class.forName0",
      "java.lang.Class.forName",
      "java.util.concurrent.FutureTask.run",
      "java.util.concurrent.ThreadPoolExecutor.runWorker",
      "java.util.concurrent.ThreadPoolExecutor$Worker.run",
      "java.lang.Thread.run"
    ]
  ]
}
```

- `id` - unique id of stacktrace
- `timestamp` - timestamp of error occurring
- `class` - classes of thrown exceptions
- `frames` - one element list of list of method calls

Of course, this stacktrace is not real :)
