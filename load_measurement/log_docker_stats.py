import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path
import subprocess as sp
import typing as t
import time


def parse_docker_stats() -> t.Dict[str, str]:
    commands = [*'docker stats --no-stream --format'.split(), 'table {{.Name}}\t{{.CPUPerc}}']
    stdout = sp.run(commands, encoding='utf-8', stdout=sp.PIPE).stdout

    result = dict(timestamp=str(time.time()))
    for line in stdout.strip().split('\n')[1:]:
        items = line.split()
        result[items[0]] = items[1].replace('%', '')

    return result


def default_containers() -> t.List[str]:
    commands = 'docker ps -a --format {{.Names}}'.split()
    return sp.run(commands, encoding='utf-8', stdout=sp.PIPE).stdout.strip().split('\n')


def default_csv_name() -> str:
    return f'docker_stats_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default='log')
    parser.add_argument('--fname', default=default_csv_name())
    parser.add_argument('--secs', type=int, default=0)
    parser.add_argument('--mins', type=int, default=0)
    parser.add_argument('--containers', '-c', nargs='+')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.dir.mkdir(exist_ok=True)
    inf_loop = (args.secs == args.mins == 0)
    until = datetime.now() + timedelta(seconds=args.secs, minutes=args.mins)

    if not args.containers:
        args.containers = default_containers()

    with args.dir.joinpath(args.fname).open('w') as f:
        fieldnames = ['timestamp'] + args.containers
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        try:
            while inf_loop or datetime.now() < until:
                writer.writerow(parse_docker_stats())

        except KeyboardInterrupt:
            print('Process is interrupted by keyboard.')
        else:
            print('Done.')


if __name__ == '__main__':
    main()
