from torch.utils.data.dataset import Dataset
# https://www.zhihu.com/column/c_1316816403623084032

'''
pytorch的 Dataset有两种：
（1） dataset.Dataset --> map-style dataset: 通过__getitem__ 来获取数据,常用的就是这个
（2） 在1.3.1之后的某个版本引入了第二种dataset -- Iterable-style datasets --> dataset.IterableDataset --> 通过__iter__得到迭代器，
        外部调用next来获得数据

'''