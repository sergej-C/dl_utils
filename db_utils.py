#- utf8 -
import pymysql.cursors
import pandas as pd


class DbUtils:

    con = None
    def __init__(self, conf):

        self.conf = conf
        self.connect()

    def connect(self):
        host = self.conf['host']
        user = self.conf['user']
        password = self.conf['password']
        db = self.conf['db']
        try:
            self.con = pymysql.connect(host=host,
                                       user=user,
                                       password=password,
                                       db=db,
                                       cursorclass=pymysql.cursors.DictCursor
                                       )
        except Exception as e:
            print "error connecting to db, {}".format(e)

    def get_pd_frame(self):
        self.encure_connection()
        with self.con.cursor() as cur:
         cur.execute("select * from `bbox_selection`")
         rows = cur.fetchall()

         df = pd.DataFrame( [[ij for ij in i] for i in rows] )
         #df.rename(columns={0: 'random Number One', 1: 'Random Number Two', 2: 'Random Number Three'}, inplace=True);

        return rows

    def exec_sql(self, sql):
        self._encure_connection()
        with self.con.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

        return rows


    def _encure_connection(self):
        if self.con is None:
            self.connect()

    def close_connection(self):

        if self.con is not None:
            try:
                self.con.close()
            except Exception as e:
                print "error on closing connecion, {}".format(e)

if __name__ == '__main__':

    conf = {
        'host':'localhost',
        'user' :'guest',
        'password':'guest',
        'db' : 'fish'
    }

    dbu = DbUtils(conf)

    print(dbu.exec_sql("select * from bbox_selection"))