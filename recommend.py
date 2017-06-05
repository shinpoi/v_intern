# -*- coding: utf-8 -*-
# Python3.5

from optparse import OptionParser
import logging

import setting
import func

parser = OptionParser()

parser.add_option("-t", "--test",
                  action="store_true", dest="t", default=False,
                  help="ランダム二次元データでk-means()ファンクションの効果をてすとする。結果が'./k_means_test.png'に保存します。")

parser.add_option("-k", "--k",
                  dest="k", default=10,
                  help="k-meansする時、グルプの数。デフォルト値は10")

parser.add_option("-u", "--user",
                  dest="u", default='(1, 345, 579, 900)',
                  help="推薦を求めっるユーザーのID。こういう形で入力します：(1, 345, 579, 900)（かっこが必要）")

options, args = parser.parse_args()

logging.info("recommend.py start")
k = int(options.k)

if __name__ == "__main__":
    if options.t:
        # def test_2d_data(data_len=600, k=10, save_name="k_means_test.png")
        func.test_2d_data(k=k)

    else:
        users = [int(i) for i in options.u[1:-1].split(',')]
        # def test_real_data(data_name='data.npy', k=10, rec_users=(1, 345, 579, 900), use_cache=True)
        func.test_real_data(k=k, rec_users=users)
