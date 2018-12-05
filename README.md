# NewWordDiscover
基于支持度 和 信息熵的新词发现算法实现

#   新词发现算法
## 使用方法
输入是一个list类型的数据，输出为一个字典类型，字典的key值为新词，
value值为该词在语料库中出现的频率。

```python
    data = 'a b c d'
    data = data.split()
    start_time = time.clock()
    newWordDiscover = NewWordDiscover()
    result = newWordDiscover.start_discover(data)
    end_time = time.clock()
    Logger.info("cost time {0}".format(end_time - start_time))
    print(result)
```
