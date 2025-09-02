import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='モデルのバリエーション')
    parser.add_argument('--NUM_POINT_PER_PATCH', help='各パッチの点数',type = int)
    parser.add_argument('-n', '--number', type=int, help='数値', required=False, default=10)
    parser.add_argument('positional', nargs='*', help='位置引数', default=[])

    args = parser.parse_args()

    # 引数を辞書型で取得
    args_dict = vars(args)

    print("引数辞書:", args_dict)

    # 辞書型から値を取り出して使用
    file_path = args_dict['file']
    number = args_dict['number']
    positional_args = args_dict['positional']

    print("ファイルパス:", file_path)
    print("数値:", number)
    print("位置引数:", positional_args)

if __name__ == "__main__":
    main()