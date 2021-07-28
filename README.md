# TraceSim
TraceSim is a method for measuring stacktrace similarity.

We describe details of algorithm in our MaLTeSQuE 2020 [paper](https://dl.acm.org/doi/10.1145/3416505.3423561) "TraceSim: A Method for Calculating Stack Trace Similarity", which can be found here: https://arxiv.org/abs/2009.12590.

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
See example of report file with stacktrace below.
Expected that report files have name like `id.json`, where `id` is stacktrace id.
- `labels_path` is a csv file with header `rid1,rid2,label`.
- `method` mean name of the method for measure similarity. 
One of the lerch, moroo, rebucket, cosine, levenshtein, brodie, prefix. 
Also, you can pass `all` for all methods comparison.

### Report example

```json
{
  "id": 123,
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

- `id` - unique id of report
- `timestamp` - timestamp of error occurring
- `class` - classes of thrown exceptions
- `frames` - one element list of list of method calls

This file should located in `stacktrace_dir` and have name `123.json`

Of course, this report is not real :)

### Labels example

The file with labels contains information about whether two reports are is similar or not.
We consider reports similar if they belong to the same group, and different otherwise.

Thus, for reports 123 and 124 from the same group, `123,124,1` will be written in the file. 
For report 125 from another group, `123,125,0` will be recorded. 
Also, by definition, it is correct to write `123,123,1`, but you should not do this.

In this case, the label file may look like this
```
rid1,rid2,label
123,124,1
123,125,0
124,125,0
```
