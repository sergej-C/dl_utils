class db_record():

    row=None
    def __init__(self, fetched_row):
        self.row=fetched_row


    def __getattr__(self, item):
        if self.row is not None:
            return self.row[item]

    def __str__(self):
        _str = ''
        for k in self.row.keys():
            _str += str(k) + ': '+ str(self.__getattr__(k))+'\n'
        return _str

    def to_list(self):
        return self.row

    def get_keys(self):
        if self.row is not None:
            return self.row.keys()
