import argparse
import csv
from datetime import datetime, timedelta
from enum import auto, Enum
import itertools
from pathlib import Path
import typing as t
import time


def parse_load_average() -> t.List[str]:
    with Path('/proc/loadavg').open('r') as f:
        vals = f.readline().strip().split()
        return vals[:3]


class CpuUsageType(Enum):
    user = 0
    nice = auto()
    system = auto()
    idle = auto()
    iowait = auto()
    irq = auto()
    softirq = auto()
    steal = auto()
    guest = auto()
    guest_nice = auto()
    usage_all = auto()


class CpuUsage:
    prev_counts: t.Dict[str, t.List[int]]

    def __init__(self):
        self.prev_counts = self.__parse_proc_stat()

    def cpu_usage_ratio(self) -> t.Dict[str, t.List[float]]:
        curr_counts = self.__parse_proc_stat()

        result = {}
        for key, prev_count in self.prev_counts.items():
            count_diff = [curr - prev for prev, curr in zip(prev_count, curr_counts[key])]
            total = sum(count_diff)
            result[key] = [diff / total for diff in count_diff]
            usage_total = total - count_diff[CpuUsageType.idle.value] - count_diff[CpuUsageType.iowait.value]
            result[key].append(usage_total / total)

        self.prev_counts = curr_counts

        return result

    @classmethod
    def __parse_proc_stat(cls) -> t.Dict[str, t.List[int]]:
        result = {}

        for line in cls.__read_proc_stat():
            if not line.startswith('cpu'):
                break
            items = line.strip().split()
            assert len(items) == len(CpuUsageType)

            result[items[0]] = [int(v) for v in items[1:]]

        return result

    @classmethod
    def get_cpu_keys(cls) -> t.List[str]:
        lines = cls.__read_proc_stat()
        return [line.strip().split()[0] for line in lines if line.startswith('cpu')]

    @staticmethod
    def __read_proc_stat() -> t.List[str]:
        with Path('/proc/stat').open('r') as f:
            return f.readlines()


DUMP_TYPES: t.List[CpuUsageType] = [
    CpuUsageType.user,
    CpuUsageType.system,
    CpuUsageType.iowait,
    CpuUsageType.irq,
    CpuUsageType.softirq,
    CpuUsageType.usage_all,
]


def make_header(cpu_keys: t.List[str]) -> t.List[str]:
    header = ['timestamp']
    header.extend(f'load_average/{i}' for i in (1, 5, 15))
    for cpu_key, type_ in itertools.product(cpu_keys, DUMP_TYPES):
        header.append(f'{cpu_key}/{type_.name}')

    return header


def make_row(cpu_keys: t.List[str], cpu_usage: CpuUsage) -> t.List[str]:
    row = [str(time.time())]
    row.extend(parse_load_average())

    usage_ratio = cpu_usage.cpu_usage_ratio()
    for cpu_key, type_ in itertools.product(cpu_keys, DUMP_TYPES):
        ratio = str(round(usage_ratio[cpu_key][type_.value] * 100, 1))  # convert to %
        row.append(ratio)

    return row


def default_csv_name() -> str:
    return f'proc_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=Path, default='log')
    parser.add_argument('--fname', default=default_csv_name())
    parser.add_argument('--secs', type=int, default=0)
    parser.add_argument('--mins', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.dir.mkdir(exist_ok=True)
    inf_loop = (args.secs == args.mins == 0)
    until = datetime.now() + timedelta(seconds=args.secs, minutes=args.mins)

    cpu_keys = CpuUsage.get_cpu_keys()
    cpu_usage = CpuUsage()

    with args.dir.joinpath(args.fname).open('w') as f:
        writer = csv.writer(f)
        writer.writerow(make_header(cpu_keys))

        try:
            while inf_loop or datetime.now() < until:
                time.sleep(1)
                writer.writerow(make_row(cpu_keys, cpu_usage))
        except KeyboardInterrupt:
            print('Process is interrupted by keyboard.')
        else:
            print('Done.')


if __name__ == '__main__':
    main()
