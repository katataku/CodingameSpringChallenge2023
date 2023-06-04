from __future__ import annotations
from collections import deque
import sys
from collections import defaultdict
import time
from collections.abc import Callable
from functools import reduce

EGG = 1
CRYSTAL = 2


class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.parents = [-1] * n

    def find(self, x: int):
        if self.parents[x] < 0:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x: int, y: int):
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.parents[x] > self.parents[y]:
            x, y = y, x

        self.parents[x] += self.parents[y]
        self.parents[y] = x

    def size(self, x: int):
        return -self.parents[self.find(x)]

    def same(self, x: int, y: int):
        return self.find(x) == self.find(y)

    def members(self, x: int):
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]

    def roots(self):
        return [i for i, x in enumerate(self.parents) if x < 0]

    def group_count(self):
        return len(self.roots())

    def all_group_members(self) -> defaultdict[int, list[int]]:
        group_members: defaultdict[int, list[int]] = defaultdict(list)
        for member in range(self.n):
            group_members[self.find(member)].append(member)
        return group_members

    def __str__(self):
        return "\n".join(f"{r}: {m}" for r, m in self.all_group_members().items())


class Cell(object):
    index: int
    cell_type: int
    resources: int
    neighbors: list[int]
    my_ants: int
    opp_ants: int
    dist_from_base: int

    def __init__(
        self,
        index: int,
        cell_type: int,
        resources: int,
        neighbors: list[int],
        my_ants: int,
        opp_ants: int,
        dist_from_base: int,
    ):
        self.index = index
        self.cell_type = cell_type
        self.resources = resources
        self.neighbors = neighbors
        self.my_ants = my_ants
        self.opp_ants = opp_ants
        self.dist_from_base = dist_from_base


class Instructions(object):
    _actions: list[str]

    def __init__(self):
        self._actions: list[str] = []

    def beacon(self, cellIdx: int, strength: int):
        return "BEACON " + str(cellIdx) + " " + str(strength)

    def line(self, sourceIdx: int, targetIdx: int, strength: int) -> str:
        return "LINE " + str(sourceIdx) + " " + str(targetIdx) + " " + str(strength)

    def msg(self, txt: str) -> str:
        return "MESSAGE " + txt

    def add_line(self, sourceIdx: int, targetIdx: int, strength: int):
        self.add(self.line(sourceIdx, targetIdx, strength))

    def add_beacon(self, cellIdx: int, strength: int):
        self.add(self.beacon(cellIdx, strength))

    def add_msg(
        self,
        txt: str,
    ):
        self.add(self.msg(txt))

    def add(self, instruction: str):
        self._actions.append(instruction)

    def print(self):
        if len(self._actions) == 0:
            debug("NO ACTION:DIRECT LINE", 1)
            for i in sorted(
                init_crystal_list, key=lambda idx: (dist[my_bases[0]][idx])
            ):
                if has_resource(i):
                    self._actions.append(
                        self.line(my_bases[0], i, MIDDLE_ANT_PROPORTION)
                    )
        print(";".join(self._actions))


def debug(txt: str, indent: int = 0):
    print("  " * indent, end="", file=sys.stderr, flush=True)
    print(txt, file=sys.stderr, flush=True)


def print_value(_str: str, indent: int = 0):
    if type(eval(_str)) in [list, dict, set, deque]:
        debug("{}:(len:{}) {}".format(_str, len(eval(_str)), eval(_str)), indent=indent)
    elif type(eval(_str)) is float:
        debug("{}: {:.2f}".format(_str, eval(_str)), indent=indent)
    else:
        debug("{}: {}".format(_str, eval(_str)), indent=indent)


def print_values(_strs: list[str], indent: int = 0):
    for str in _strs:
        print_value(str, indent=indent)


def print_game_tactic():
    debug(f"game_tactic: {game_tactic} ({game_tactic_dict[game_tactic]})", 1)


# start input
cells: list[Cell] = []
number_of_cells = int(input())  # amount of hexagonal cells in this map

# 時間計測開始
time_sta = time.perf_counter()
debug(f"===init start===", 0)


INF = number_of_cells + 1
# distance matrix
dist: list[list[int]] = []
for _ in range(number_of_cells):
    dist.append([INF for _ in range(number_of_cells)])

for i in range(number_of_cells):
    dist[i][i] = 0


initial_resources_total: int = 0
init_crystal_list: list[int] = []
init_egg_list: list[int] = []
init_crystal_total: int = 0
init_egg_total: int = 0

for i in range(number_of_cells):
    inputs = [int(j) for j in input().split()]
    cell_type = inputs[0]  # 0 for empty, 1 for eggs, 2 for crystal
    if cell_type == CRYSTAL:
        init_crystal_list.append(i)
    if cell_type == EGG:
        init_egg_list.append(i)
    # the initial amount of eggs/crystals on this cell
    initial_resources = inputs[1]
    init_crystal_total += initial_resources if cell_type == CRYSTAL else 0
    init_egg_total += initial_resources if cell_type == EGG else 0
    # the index of the neighbouring cell for each direction
    neigh_0 = inputs[2]
    neigh_1 = inputs[3]
    neigh_2 = inputs[4]
    neigh_3 = inputs[5]
    neigh_4 = inputs[6]
    neigh_5 = inputs[7]
    c: Cell = Cell(
        index=i,
        cell_type=cell_type,
        resources=initial_resources,
        neighbors=list(
            filter(
                lambda id: id > -1,
                [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5],
            )
        ),
        my_ants=0,
        opp_ants=0,
        dist_from_base=-1,
    )
    cells.append(c)
    for next in c.neighbors:
        dist[i][next] = 1

initial_resources_total = init_crystal_total + init_egg_total
VICTORY_THRESHOLD: int = (init_crystal_total) // 2
number_of_bases: int = int(input())
my_bases: list[int] = []
for i in input().split():
    my_base_index = int(i)
    my_bases.append(my_base_index)
opp_bases: list[int] = []
for i in input().split():
    opp_base_index = int(i)
    opp_bases.append(opp_base_index)

for k in range(number_of_cells):
    for i in range(number_of_cells):
        for j in range(i, number_of_cells):
            dist[i][j] = min(
                dist[i][j],
                dist[i][k] + dist[k][j],
            )
            dist[j][i] = dist[i][j]

DISTANCE = 0
INDEX = 1


# get_nearest
# src_idxからtar_listの中で最も近いindexを返す
# 返り値は(tar_listの中で最も近い距離, tar_listの中で最も近いindex)
def get_nearest(src_idx: int, tar_list: list[int]) -> tuple[int, int]:
    global dist
    my_lambda: Callable[[int], tuple[int, int]] = lambda idx: (dist[src_idx][idx], idx)
    return min([my_lambda(tar) for tar in tar_list])


def get_nearest_my_base(idx: int) -> tuple[int, int]:
    global my_bases
    return get_nearest(idx, my_bases)


def get_nearest_opp_base(idx: int) -> tuple[int, int]:
    global opp_bases
    return get_nearest(idx, opp_bases)


def has_resource(idx: int) -> bool:
    return cells[idx].resources > 0


def is_egg(idx: int) -> bool:
    return cells[idx].cell_type == EGG


def is_crystal(idx: int) -> bool:
    return cells[idx].cell_type == CRYSTAL


def is_egg_with_resource(idx: int) -> bool:
    return is_egg(idx) and has_resource(idx)


def is_crystal_with_resource(idx: int) -> bool:
    return is_crystal(idx) and has_resource(idx)


acc = 0
visit_crystal_list: list[int] = []
middle_crystal_list: list[int] = []
my_close_crystal_list: list[int] = []
my_half_crystal_list: list[int] = []
for i in sorted(
    init_crystal_list,
    key=lambda idx: (
        get_nearest_my_base(idx)[DISTANCE],
        -1 * get_nearest_opp_base(idx)[DISTANCE],
    ),
):
    index = cells[i].index
    nearest_base = get_nearest_my_base(index)[INDEX]
    nearest_opp_base_distance = get_nearest_opp_base(index)[DISTANCE]
    if dist[nearest_base][index] <= nearest_opp_base_distance:
        my_half_crystal_list.append(i)
    if -2 <= (dist[nearest_base][index] - nearest_opp_base_distance) <= 1:
        middle_crystal_list.append(i)
    if dist[nearest_base][index] * 3 < nearest_opp_base_distance:
        my_close_crystal_list.append(i)
    acc += cells[i].resources
    if init_crystal_total * 0 <= acc * 100 <= init_crystal_total * 60:
        visit_crystal_list.append(i)

visit_crystal_list = middle_crystal_list.copy()

print_values(["init_crystal_list", "visit_crystal_list"], 1)
init_egg_list.sort(
    key=lambda idx: (
        get_nearest_my_base(idx)[DISTANCE],
        -1 * get_nearest_opp_base(idx)[DISTANCE],
    )
)
visit_egg_list: list[int] = []
acc = 0
for i in init_egg_list:
    if (
        init_egg_total * 0 <= acc * 100 <= init_egg_total * 49
        and get_nearest_opp_base(i)[DISTANCE] > 1
    ):
        visit_egg_list.append(i)
        acc += cells[i].resources

print_value("init_egg_list", 1)
print_value("visit_egg_list", 1)
game_tactic_dict = {
    0: "Early Game",
    1: "Mid Game",
    2: "Late Game",
    10: "only one crystal",
}
game_tactic = 0


# 時間計測終了
time_end = time.perf_counter()
# 経過時間（秒）
debug("init time[ms]:" + str((time_end - time_sta) * 1000), 1)
debug(f"===init end===", 0)

TINY_ANT_PROPORTION: int = 1
LOW_ANT_PROPORTION: int = 4
MIDDLE_ANT_PROPORTION: int = 5

progress_indicator = 0
egg_list: list[int] = []
crystal_list: list[int] = []
crystal_resource_total = 0
egg_resource_total = 0
my_ants_total = 0
my_score: int = 0
opp_score: int = 0


def turn_input():
    global my_score, opp_score, egg_list, crystal_list, crystal_resource_total, egg_resource_total, my_ants_total

    inputs = [int(j) for j in input().split()]
    my_score = inputs[0]
    opp_score = inputs[1]

    egg_list = []
    crystal_list = []
    crystal_resource_total = 0
    egg_resource_total = 0
    my_ants_total = 0
    for i in range(number_of_cells):
        inputs = [int(j) for j in input().split()]
        resources = inputs[0]  # the current amount of eggs/crystals on this cell
        my_ants = inputs[1]  # the amount of your ants on this cell
        opp_ants = inputs[2]  # the amount of opponent ants on this cell

        cells[i].resources = resources
        cells[i].my_ants = my_ants
        cells[i].opp_ants = opp_ants
        cell_type = cells[i].cell_type

        my_ants_total += my_ants
        crystal_resource_total += resources if cell_type == CRYSTAL else 0
        egg_resource_total += resources if cell_type == EGG else 0

        if is_egg_with_resource(i):
            egg_list.append(i)
        if is_crystal_with_resource(i):
            crystal_list.append(i)


def calc_progress_indicator():
    global progress_indicator
    progress_indicator = 1 - (
        (egg_resource_total + crystal_resource_total) / initial_resources_total
    )
    print_values(["progress_indicator"], 1)


def update_game_tactic():
    debug("===update_game_tactic===", 0)
    global game_tactic, nearest_resource_list
    if game_tactic == 0:
        if progress_indicator >= 0.3:
            debug("===__trug__ progress_indicator >= 0.3===", 1)
            game_tactic = 1
        else:
            nearest_resource_list = sorted(
                [x for x in crystal_list + egg_list if has_resource(x)],
                key=lambda idx: (
                    get_nearest_my_base(idx)[DISTANCE],
                    cells[idx].cell_type,
                ),
            )
            print_values(["nearest_resource_list"], 1)
            nearest_resource_idx = nearest_resource_list[0]
            nearest_resource_dist = get_nearest_my_base(nearest_resource_idx)[DISTANCE]
            if not (
                is_egg_with_resource(nearest_resource_idx)
                and nearest_resource_dist <= 2
            ):
                game_tactic = 1
    if len(crystal_list) == 1 and my_ants_total > cells[crystal_list[0]].resources:
        game_tactic = 10


# TODO: ループの所要時間を計測して、出力する→最終版では削除する
# game loop
while True:
    # 時間計測開始
    time_sta = time.perf_counter()

    # Input Game State
    turn_input()
    nearest_resource_list: list[int] = []

    # A. Pre processing
    calc_progress_indicator()
    update_game_tactic()

    # B. Game strategy
    inst = Instructions()
    #  if nearest is egg, go to egg
    if game_tactic == 0:
        print_game_tactic()
        nearest_resource_idx = nearest_resource_list[0]
        nearest_resource_dist = get_nearest_my_base(nearest_resource_idx)[DISTANCE]

        tar_base_set: set[int] = set()
        nearest_resources_amount = 0
        nearest_resource_path_way_long = 0
        for target_resource in nearest_resource_list:
            if not is_egg(target_resource):
                break
            tar_dist, tar_base = get_nearest_my_base(target_resource)
            if tar_dist != nearest_resource_dist:
                break
            inst.add_line(
                tar_base,
                target_resource,
                TINY_ANT_PROPORTION,
            )

            tar_base_set.add(tar_base)
            nearest_resources_amount += cells[target_resource].resources
            nearest_resource_path_way_long += tar_dist
            for neighbor in [
                x for x in cells[target_resource].neighbors if is_egg_with_resource(x)
            ]:
                inst.add_beacon(neighbor, TINY_ANT_PROPORTION)
                nearest_resources_amount += cells[neighbor].resources
                nearest_resource_path_way_long += 1
        if nearest_resources_amount * (
            nearest_resource_path_way_long + 1
        ) > my_ants_total and len(tar_base_set) == len(my_bases):
            inst.print()
            continue

        # my ants are enough to get all eggs
        debug("TARGET CHANGE: enough to egg or multi base", 1)
        game_tactic = 1

    # main strategy
    if game_tactic == 1:
        print_game_tactic()
        visit_crystal_list: list[int] = [
            x for x in visit_crystal_list if has_resource(x)
        ]
        visit_egg_list: list[int] = [x for x in visit_egg_list if has_resource(x)]
        visit_resource_candidates: list[int] = [
            x
            for x in visit_crystal_list + visit_egg_list
            if get_nearest_my_base(x)[DISTANCE] * 1.5 < my_ants_total
        ]

        if len(visit_resource_candidates) <= 0 or len(visit_crystal_list) <= 2:
            debug("no middle resource to visit, visit all", 1)
            init_crystal_list: list[int] = [
                x for x in init_crystal_list if has_resource(x)
            ]
            visit_resource_candidates = sorted(
                init_crystal_list + visit_egg_list,
                key=lambda idx: (
                    (
                        get_nearest_my_base(idx)[DISTANCE]
                        - get_nearest_opp_base(idx)[DISTANCE]
                    ),
                    get_nearest_my_base(idx)[DISTANCE],
                ),
            )

        visit_resource_list: list[int] = []
        acc = 0
        for cell in [x for x in visit_resource_candidates if is_egg(x)]:
            visit_resource_list.append(cell)
        for cell in [x for x in visit_resource_candidates if is_crystal(x)]:
            if acc + my_score < VICTORY_THRESHOLD:
                visit_resource_list.append(cell)
                acc += cells[cell].resources
        print_values(["acc + my_score", "VICTORY_THRESHOLD"], 1)

        visit_resource_list.sort(
            key=lambda idx: (
                get_nearest_my_base(idx)[DISTANCE],
                cells[idx].cell_type,
                cells[idx].resources * -1,
            )
        )
        print_values(["visit_crystal_list", "visit_egg_list", "visit_resource_list"], 1)
        # TODO:過剰にantを送らないようにする。
        # idea:connected_to_baseを[int,int]に、[idx,max_resource_num]として最大値を管理。
        # idea:もしかしたらqueで管理した方がいいかも。
        # idea:あと、早めにclose_cellに向かう判定もつけるべき？（全ての候補が過剰なケースもありそう）
        connected_to_base = my_bases.copy()
        que: deque[int] = deque(my_bases + visit_resource_list)
        uf = UnionFind(number_of_cells)

        history_dict: dict[int, int] = {}
        debug(f"===loop start===", 1)

        while len(que) > 0:
            debug(f"----new loop---", 1)
            current_pos_idx: int = que.popleft()
            print_values(
                [
                    "current_pos_idx",
                    "connected_to_base",
                    "que",
                ],
                2,
            )

            # if current_pos_idx is already connected to base, skip
            if current_pos_idx in connected_to_base:
                # if neighbor has resource, go to neighbor
                for neighbor in cells[current_pos_idx].neighbors:
                    if (
                        is_egg_with_resource(neighbor)
                        and get_nearest_opp_base(neighbor)[DISTANCE] > 1
                        and not uf.same(current_pos_idx, neighbor)
                    ):
                        que.appendleft(neighbor)
                        uf.union(current_pos_idx, neighbor)
                        connected_to_base.append(neighbor)
                continue

            def get_destination_candidates(current_pos_idx: int) -> list[int]:
                global connected_to_base
                global uf
                global cells
                return sorted(
                    [x for x in connected_to_base if not uf.same(x, current_pos_idx)],
                    key=lambda x: (
                        dist[current_pos_idx][x],
                        get_nearest_my_base(x)[DISTANCE],
                        cells[x].my_ants * -1,
                    ),
                )

            dest_candidates: list[int] = get_destination_candidates(current_pos_idx)
            print_value(
                "dest_candidates",
                2,
            )
            if len(dest_candidates) == 0:
                continue
            dest: int = dest_candidates[0]
            # ant resource check: ants are not enough to get the resource, discard
            if my_ants_total - len(connected_to_base) < dist[current_pos_idx][dest]:
                debug("ant resource check: ants are not enough to get the resource", 2)
                continue

            def get_next_hop_candidates(current_pos_idx: int, dest: int) -> list[int]:
                global dist
                global cells
                return sorted(
                    [
                        x
                        for x in cells[current_pos_idx].neighbors
                        if dist[dest][x] + 1 == dist[dest][current_pos_idx]
                    ],
                    key=lambda x: (
                        get_nearest_my_base(x)[DISTANCE],
                        -1 * cells[x].resources,
                        -1 * cells[x].my_ants,
                    ),
                )

            next_hop_candidates = get_next_hop_candidates(current_pos_idx, dest)

            print_values(
                [
                    "next_hop_candidates",
                    "dist[current_pos_idx][dest]",
                ],
                2,
            )

            # if neighbor has resource, go to neighbor
            # to find path in skip logic, re-append current_pos_idx to que
            for neighbor in next_hop_candidates:
                if is_egg_with_resource(neighbor):
                    que.appendleft(current_pos_idx)
                    break

            # path identified
            # or infinite-loop
            if len(next_hop_candidates) == 1 or history_dict.get(
                current_pos_idx, -1
            ) == len(que):
                if len(next_hop_candidates) == 1:
                    debug("path identified", 3)
                else:
                    debug("infinite-loop", 3)
                uf.union(current_pos_idx, next_hop_candidates[0])
                connected_to_base.append(current_pos_idx)
                que.appendleft(next_hop_candidates[0])
                continue
            history_dict[current_pos_idx] = len(que)

            # path not defined
            # - one of the next_cells is already connected to base
            # - one of the next_cells is egg, put it on to the que
            for one_of_next_cell in next_hop_candidates:
                if one_of_next_cell in connected_to_base:
                    uf.union(current_pos_idx, one_of_next_cell)
                if cells[one_of_next_cell].resources == 1:
                    que.appendleft(one_of_next_cell)

            # don't handle, push back to que to retry
            if not current_pos_idx in que:
                que.append(current_pos_idx)
            connected_to_base.append(current_pos_idx)
        debug(f"---loop end---", 2)
        print_value("connected_to_base", 2)

        # start to union, connect isolated islands to base
        # (isolated islands made only if path undefined)
        connected_to_base.sort(key=lambda idx: get_nearest_my_base(idx)[DISTANCE])
        verified_connection_cells = my_bases.copy()
        for current_pos_idx in connected_to_base:
            if all([not uf.same(x, current_pos_idx) for x in my_bases]):
                dest = sorted(
                    verified_connection_cells,
                    key=lambda x: (dist[current_pos_idx][x], cells[x].my_ants * -1),
                )[0]
                inst.add_line(dest, current_pos_idx, MIDDLE_ANT_PROPORTION)
                uf.union(dest, current_pos_idx)
            verified_connection_cells.append(current_pos_idx)

        print_value("connected_to_base", 2)

        for cell in connected_to_base:
            if cell in my_bases:
                if any([x in connected_to_base for x in cells[cell].neighbors]):
                    inst.add_beacon(cell, MIDDLE_ANT_PROPORTION)
            else:
                inst.add_beacon(cell, MIDDLE_ANT_PROPORTION)

    if game_tactic == 10:
        print_game_tactic()
        acc: int = 0
        for cell in crystal_list:
            inst.add_line(
                cell,
                get_nearest_my_base(cell)[1],
                TINY_ANT_PROPORTION,
            )
            acc += cells[cell].resources
            if acc + my_score > VICTORY_THRESHOLD:
                break
        last_crystal_idx = crystal_list[0]

    inst.print()

    # 時間計測終了
    time_end = time.perf_counter()
    # 経過時間（秒）
    debug("loop time[ms]:" + str((time_end - time_sta) * 1000))
